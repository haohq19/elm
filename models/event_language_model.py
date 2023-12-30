from typing import List, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging

from models.event_transformer import EventFrameEmbedding
from models.perceiver_resampler import PerceiverResampler

proxies={'http': 'http://127.0.0.1:15777', 'https': 'http://127.0.0.1:15777'}

logging.set_verbosity_error()
logger = logging.get_logger('transformers')


def accumulate_padding(input_embeds: torch.Tensor, attention_mask: torch.Tensor, padding_side: str = 'right'):
    assert padding_side in ['right', 'left']

    new_input_embeds = torch.empty_like(input_embeds)
    new_attention_masks = torch.empty_like(attention_mask)

    for i, (embed, mask) in enumerate(zip(input_embeds, attention_mask)):
        padding_indices = torch.where(mask == 0)[0]
        non_padding_indices = torch.where(mask == 1)[0]
        if padding_side == 'left':
            new_indices = torch.cat((padding_indices, non_padding_indices), dim=0)
        else:
            new_indices = torch.cat((non_padding_indices, padding_indices), dim=0)
        new_input_embeds[i] = embed.index_select(0, new_indices)
        new_attention_masks[i] = mask.index_select(0, new_indices)

    return new_input_embeds, new_attention_masks


class LanguageDecoder(nn.Module):

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        self.forward = self.model.forward
        self.generate = self.model.generate

    @property
    def model_id(self) -> str:
        return type(self.model).__name__.lower()

    @property
    def embed_dim(self) -> int:
        if 'gpt' in self.model_id:
            return self.model.config.n_embd
        elif 'opt' in self.model_id:
            return self.model.config.word_embed_proj_dim
        else:
            raise NotImplementedError

    @property
    def embed_tokens(self) -> nn.Module:
        if 'gpt' in self.model_id:
            return self.model.transformer.wte
        elif 'opt' in self.model_id:
            return self.model.model.decoder.embed_tokens
        else:
            raise NotImplementedError

    def prepare_inputs_for_generation(self, input_ids, attention_mask, visual_embeds, past_key_values=None, use_cache=None, **kwargs):
        expand_size = input_ids.size(0) // visual_embeds.size(0)
        visual_embeds = visual_embeds.repeat_interleave(expand_size, dim=0)
        visual_mask = torch.ones(visual_embeds.shape[:2], dtype=torch.long, device=visual_embeds.device)

        if input_ids[0][0] == self.model.config.bos_token_id:
            input_ids = input_ids[:, 1:]
            attention_mask = attention_mask[:, 1:]

        token_embeds = self.embed_tokens(input_ids)
        
        input_embeds = torch.cat([visual_embeds, token_embeds], dim=1)
        attention_mask = torch.cat([visual_mask, attention_mask], dim=1)

        input_embeds, attention_mask = accumulate_padding(input_embeds, attention_mask, padding_side='left')

        if past_key_values:
            input_embeds = input_embeds[:, -1].unsqueeze(1)

        return {
            'inputs_embeds': input_embeds,
            'attention_mask': attention_mask,
            'past_key_values': past_key_values,
            'use_cache': use_cache
        }


class EventLanguageModel(nn.Module):

    def __init__(self, d_vision, d_model):
        super(EventLanguageModel, self).__init__()
        self.vision_encoder = EventFrameEmbedding(in_channels=2, d_model=d_vision)
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B', proxies=proxies)
        if self.tokenizer._pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.language_decoder = LanguageDecoder(AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-j-6B', torch_dtype=torch.float16, revision='float16', low_cpu_mem_usage=True, proxies=proxies))
        for param in self.language_decoder.parameters():
            param.requires_grad = False
        
        self.mapper = PerceiverResampler(d_input=d_vision, d_model=d_model)
        
    
    def embed_event(self, events: torch.Tensor):
        # embed event frames into latent vectors
        # events.shape = [batch_size, nsteps, in_channels, height, width]
        with torch.no_grad():
            vision_embeds = self.vision_encoder(events)  # embed.shape = [batch_size, nsteps, d_vision]
        mapped_vision_embeds = self.mapper(vision_embeds)  # mapped_embed.shape = [batch_size, nsteps, d_model]
        return mapped_vision_embeds
    
    def embed_text(self, input_ids: torch.Tensor):
        with torch.no_grad():
            token_embeds = self.language_decoder.embed_tokens(input_ids)    # token_embeds.shape = [batch_size, nsteps, d_model]
        return token_embeds

    def forward(self, events, target_ids, prefix_ids = None):
        # events.shape = [batch_size, nsteps, d_event, height, width]
        visual_embeds = self.embed_event(events).half()  # visual_embed.shape = [batch_size, nsteps, d_model]
        target_embeds = self.embed_text(target_ids)  # target_embeds.shape = [batch_size, ntokens, d_model]
        if prefix_ids is None:
            input_embeds = torch.cat((visual_embeds, target_embeds), dim=1)  # input_embeds.shape = [batch_size, nsteps + ntokens, d_model]
        else:
            prefix_embeds = self.embed_text(prefix_ids)
            input_embeds = torch.cat((visual_embeds, prefix_embeds, target_embeds), dim=1)

        visual_mask = torch.ones(visual_embeds.shape[:2], dtype=torch.long, device=visual_embeds.device)
        target_token_mask = (target_ids != self.tokenizer.pad_token_id).long()
        if prefix_ids is None:
            attention_mask = torch.cat((visual_mask, target_token_mask), dim=1)
        else:
            prefix_token_mask = (prefix_ids != self.tokenizer.pad_token_id).long()
            attention_mask = torch.cat((visual_mask, prefix_token_mask, target_token_mask), dim=1)
        
        input_embeds, attention_mask = accumulate_padding(input_embeds, attention_mask, padding_side='right')

        outputs = self.language_decoder(inputs_embeds=input_embeds, attention_mask=attention_mask)
        
        return outputs
    
    def text_transform(self, text: Union[str, List[str]], **kwargs) -> torch.Tensor:
        return self.tokenizer(text, padding='longest', return_tensors='pt', **kwargs)