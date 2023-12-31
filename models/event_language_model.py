from typing import List, Union

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging

from models.event_transformer import EventFrameEmbedding
from models.perceiver_resampler import PerceiverResampler

proxies={'http': 'http://127.0.0.1:15777', 'https': 'http://127.0.0.1:15777'}

logging.set_verbosity_error()
logger = logging.get_logger('transformers')


class EventLanguageModel(nn.Module):

    def __init__(self, d_vision, d_model, lm_model_id='gpt2-large'):
        super(EventLanguageModel, self).__init__()
        self.d_vision = d_vision
        self.d_model = d_model
        self.lm_model_id = lm_model_id

        self.vision_encoder = EventFrameEmbedding(in_channels=2, d_model=d_vision)
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(lm_model_id, proxies=proxies)
        if self.tokenizer._pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.language_decoder = AutoModelForCausalLM.from_pretrained(lm_model_id, torch_dtype=torch.float32, proxies=proxies)
        for param in self.language_decoder.parameters():
            param.requires_grad = False
        
        self.mapper = PerceiverResampler(d_input=d_vision, d_model=d_model)
        
    
    def embed_event(self, events: torch.Tensor):
        # embed event frames into latent vectors
        # events.shape = [batch_size, nsteps, in_channels, height, width]
        with torch.no_grad():
            embeds = self.vision_encoder(events)  # embed.shape = [batch_size, nsteps, d_vision]
        embeds = self.mapper(embeds)  # embed.shape = [batch_size, nsteps, d_model]
        return embeds
    
    def embed_text(self, ids: torch.Tensor):
        # embed text tokens into latent vectors
        # ids.shape = [batch_size, ntokens]
        with torch.no_grad():
            embeds = self.language_decoder.transformer.wte(ids)    # token_embeds.shape = [batch_size, ntokens, d_model]
        return embeds

    def forward(self, events: int, target_ids: torch.Tensor, prefix_ids = None):
        # events.shape = [batch_size, nsteps, 2, height, width]
        # target_ids.shape = [batch_size, ntargets]
        # prefix_ids.shape = [batch_size, nprefixs]
        event_embeds = self.embed_event(events).half()  # event_embed.shape = [batch_size, nsteps, d_model]
        target_embeds = self.embed_text(target_ids)  # target_embeds.shape = [batch_size, ntargets, d_model]
        if prefix_ids is None:
            input_embeds = torch.cat((event_embeds, target_embeds), dim=1)  # input_embeds.shape = [batch_size, nsteps + ntargets, d_model]
        else:
            prefix_embeds = self.embed_text(prefix_ids)
            input_embeds = torch.cat((event_embeds, prefix_embeds, target_embeds), dim=1)  # input_embeds.shape = [batch_size, nsteps + nprefixs + ntargets, d_model]

        event_mask = torch.ones(event_embeds.shape[:2], dtype=torch.long, device=event_embeds.device)
        target_mask = (target_ids != self.tokenizer.pad_token_id).long()
        if prefix_ids is None:
            attention_mask = torch.cat((event_mask, target_mask), dim=1)
        else:
            prefix_mask = (prefix_ids != self.tokenizer.pad_token_id).long()
            attention_mask = torch.cat((event_mask, prefix_mask, target_mask), dim=1)

        outputs = self.language_decoder(inputs_embeds=input_embeds, attention_mask=attention_mask)
        
        return outputs