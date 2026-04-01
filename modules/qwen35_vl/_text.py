import torch
from transformers.models.qwen3_5 import Qwen3_5TextModel

class Qwen35VLTextModelOpt(Qwen3_5TextModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs,
    ):
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=None,
                position_ids=text_position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=None,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states