import torch
from transformers import Qwen3VLForConditionalGeneration


class Qwen3VLForConditionalGenerationOpt(Qwen3VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    def forward(
            self,
            hidden_states: torch.LongTensor = None,
            **kwargs,
        ):
        logits = self.lm_head(hidden_states[:, -1:, :])
        return logits
