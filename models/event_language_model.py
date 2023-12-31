from typing import List, Union, Tuple, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging

from models.event_transformer import EventFrameEmbedding
from models.perceiver_resampler import PerceiverResampler

proxies={'http': 'http://127.0.0.1:15777', 'https': 'http://127.0.0.1:15777'}

logging.set_verbosity_error()
logger = logging.get_logger('transformers')


def accumulate_padding(input_embeds: torch.Tensor, attention_mask: torch.Tensor, padding_side: str = 'right') -> Tuple[torch.Tensor, torch.Tensor]:
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


class EventLanguageModel(nn.Module):
    """
    EventLanguageModel is a PyTorch module that combines event frame embedding and text token embedding
    to generate language representations for event understanding tasks.

    Args:
        d_vision (int): The dimension of the event frame embedding.
        d_model (int): The dimension of the language representation.
        lm_model_id (str, optional): The identifier of the pre-trained language model. Defaults to 'gpt2-large'.

    Attributes:
        vision_encoder (EventFrameEmbedding): The event frame embedding module.
        tokenizer (AutoTokenizer): The tokenizer for text token embedding.
        language_decoder (AutoModelForCausalLM): The language model for text token embedding.
        mapper (PerceiverResampler): The module for mapping event frame embedding to language representation.

    Methods:
        embed_event(events: torch.Tensor) -> torch.Tensor:
            Embeds event frames into latent vectors.

        embed_text(ids: torch.Tensor) -> torch.Tensor:
            Embeds text tokens into latent vectors.

        forward(events: int, target_ids: torch.Tensor, prefix_ids=None) -> torch.Tensor:
            Forward pass of the EventLanguageModel.

    """

    def __init__(self, d_vision: int, d_model: int, lm_model_id: str = 'gpt2-large'):
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
        
    
    def embed_event(self, events: torch.Tensor) -> torch.Tensor:
        """
        Embeds event frames into latent vectors.

        Args:
            events (torch.Tensor): The event frames. Shape: [batch_size, nsteps, in_channels, height, width]

        Returns:
            torch.Tensor: The embedded event frames. Shape: [batch_size, nsteps, d_model]
        """
        with torch.no_grad():
            embeds = self.vision_encoder(events)  # embed.shape = [batch_size, nsteps, d_vision]
        embeds = self.mapper(embeds)  # embed.shape = [batch_size, nsteps, d_model]
        return embeds
    
    def embed_text(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Embeds text tokens into latent vectors.

        Args:
            ids (torch.Tensor): The text tokens. Shape: [batch_size, ntokens]

        Returns:
            torch.Tensor: The embedded text tokens. Shape: [batch_size, ntokens, d_model]
        """
        with torch.no_grad():
            embeds = self.language_decoder.transformer.wte(ids)    # token_embeds.shape = [batch_size, ntokens, d_model]
        return embeds

    def forward(self, events: int, target_ids: torch.Tensor, prefix_ids : Union(Optional, torch.Tensor) = None) -> torch.Tensor:
        """
        Forward pass of the EventLanguageModel.

        Args:
            events (int): The event frames. Shape: [batch_size, nsteps, 2, height, width]
            target_ids (torch.Tensor): The target text tokens. Shape: [batch_size, ntargets]
            prefix_ids (torch.Tensor, optional): The prefix text tokens. Shape: [batch_size, nprefixs]
        """
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

        input_embeds, attention_mask = accumulate_padding(input_embeds, attention_mask, padding_side='right')
        
        start_idx = event_mask.sum(dim=1)
        if prefix_ids is not None:
            start_idx += prefix_mask.sum(dim=1)
        end_idx = start_idx + target_mask.sum(dim=1)
        labels = torch.full_like(attention_mask, -100)  # -100 is the ignore index for cross-entropy loss
        for i, (j, k) in enumerate(zip(start_idx, end_idx)):
            labels[i, j:k] = target_ids[i, :k-j]
        
        outputs = self.language_decoder(inputs_embeds=input_embeds, attention_mask=attention_mask, labels=labels)
        
        return outputs
