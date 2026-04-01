import torch
from transformers import Qwen3_5ForConditionalGeneration


class Qwen35VLForConditionalGenerationOpt(Qwen3_5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    def forward(
            self,
            hidden_states: torch.LongTensor = None,
            **kwargs,
        ):
        logits = self.lm_head(hidden_states[:, -1:, :])
        return logits
