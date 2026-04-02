import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
from typing import Any, Callable, Optional, Union
import itertools


class Qwen35VLModelOpt(Qwen3_5Model):
    def __init__(self, qwen_config, onnx_config):
        self.batch_size = onnx_config.batch_size
        self.imgs_nums = len(onnx_config.imgs_paths)
        super().__init__(qwen_config)

    def get_rope_index(
        self,
        input_ids: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        spatial_merge_size = self.config.vision_config.spatial_merge_size

        mrope_position_deltas = []
        position_ids = torch.zeros(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )

        for batch_idx in range(self.batch_size):
            current_input_ids = input_ids[batch_idx]
            input_token_type = mm_token_type_ids[batch_idx]
            grid_thw = image_grid_thw[batch_idx]

            current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]
            input_token_type = input_token_type[attention_mask[batch_idx].bool()]

            visual_mask = (input_token_type == 1).float()
            first_one = visual_mask.argmax(dim=0).item()
            last_one = int((visual_mask * torch.arange(visual_mask.size(0), device=visual_mask.device)).max().item())

            llm_pos_ids_list = []
            # llm part
            llm_pos_ids_list.append(
                torch.arange(first_one, device=input_ids.device).view(1, -1).expand(3, -1)
            )

            # vision part
            vision_position_ids = self.get_vision_position_ids(
                first_one, grid_thw, 1, spatial_merge_size, device=input_ids.device
            )
            llm_pos_ids_list.append(vision_position_ids)
            first_one += max(grid_thw[1], grid_thw[2]) // spatial_merge_size

            # llm part
            llm_pos_ids_list.append(
                torch.arange(input_token_type.size(0) - last_one - 1, device=input_ids.device).view(1, -1).expand(3, -1) + first_one
            )

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)

            position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas

    def get_image_features(self, image_embeds: torch.Tensor, **kwargs):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model. The deepstack visual features are also returned.

        Args:
            image_embeds (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
        """
        # image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        # split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, 64)
        return image_embeds


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        inputs_embeds = self.get_input_embeddings()(input_ids) # torch.Size([1, 144, 2048])

        # process image use vit model
        image_embeds = self.get_image_features(image_embeds)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        vision_positions, rope_deltas = self.get_rope_index(
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )
        # self.model.rope_deltas = rope_deltas

        text_positions = attention_mask.long().cumsum(-1) - 1
        # We need this as otherwise padding tokens appear as -1 in position
        text_positions = text_positions.masked_fill(attention_mask == 0, 0)
        text_positions = text_positions[None, ...]
        position_ids = torch.cat([text_positions, vision_positions], dim=0)

        return position_ids, inputs_embeds
