import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
from typing import Any, Callable, Optional, Union


class Qwen35VLModelOpt(Qwen3_5Model):
    def __init__(self, qwen_config, onnx_config):
        self.batch_size = onnx_config.batch_size
        self.imgs_nums = len(onnx_config.imgs_paths)
        super().__init__(qwen_config)

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Different from the original implementation, Qwen3VL use timestamps rather than absolute time position ids."""

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        mrope_position_deltas = []
        total_input_ids = input_ids

        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        # attention_mask = attention_mask.to(total_input_ids.device)
        for i in range(self.batch_size):
            input_ids = input_ids[i]
            image_nums = torch.tensor([self.imgs_nums], dtype=torch.int64)
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            st_idx = 0
            remain_images = image_nums
            for _ in range(image_nums):
                ed_image = input_tokens.index(image_token_id, st)

                t, h, w = (1, 16, 16)
                image_index += 1
                remain_images -= 1
                ed = ed_image

                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t,
                    h // spatial_merge_size,
                    w // spatial_merge_size,
                )
                text_len = ed - st

                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                # t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the temporal information for videos)
                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w
                st_idx = llm_pos_ids_list[-1].max() + 1

            text_len = input_ids.shape[0] - st
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            # position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            position_ids[:, i, :] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
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

        position_ids = self.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            mm_token_type_ids=mm_token_type_ids,
        )

        return position_ids, inputs_embeds
