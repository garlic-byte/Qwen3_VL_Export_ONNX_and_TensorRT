import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel, Qwen3VLTextModel
from typing import Any, Callable, Optional, Union


class Qwen3VLTextModelOpt(Qwen3VLTextModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # if cache_position is None:
        past_seen_tokens = 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
        text_position_ids = position_ids[0]

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # create attentions_mask by eager kind
        base_mask = torch.tril(torch.ones(position_ids.size(2), position_ids.size(2), dtype=torch.bool)).to(inputs_embeds.device)
        mask = torch.full((position_ids.size(2), position_ids.size(2)), fill_value=-3.4028e+38, dtype=torch.float32, device=inputs_embeds.device)
        mask = mask.masked_fill(base_mask, 0.0)

        # size from 144*144 -> 1*1*144*144
        attention_mask = mask.unsqueeze(0).unsqueeze(0)


        # decoder layers
        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs

            # add visual features to the hidden states of first several layers
            if layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states = self.norm(hidden_states)

        return hidden_states

    def _deepstack_process(
            self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor
    ):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)

        indices = torch.nonzero(visual_pos_masks.squeeze(0), as_tuple=False).squeeze(1)
        indices_expanded = indices.unsqueeze(0).unsqueeze(-1).expand(1, -1, hidden_states.size(2))
        new_values = hidden_states[0, indices] + visual_embeds
        hidden_states = hidden_states.scatter(1, indices_expanded, new_values.unsqueeze(0))
        return hidden_states
