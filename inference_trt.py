import os
import time

from qwen3_vl_export_onnx import get_model_input, ArgsConfig
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast
import torch
from utils import trt_torch as trt
from functools import partial


def llm_tensorrt_forward(self, position_ids, inputs_embeds, visual_pos_masks, deepstack_visual_embeds):
    print("llm_tensorrt_forward")
    print(f"{position_ids.shape}, {position_ids.dtype}, {position_ids.device}")
    print(f"{inputs_embeds.shape}, {inputs_embeds.dtype}, {inputs_embeds.device}")
    print(f"visual_pos_masks.shape: {visual_pos_masks.shape}")
    print(f"deepstack_visual_embeds.shape: {deepstack_visual_embeds.shape}")

    self.llm_engine.set_runtime_tensor_shape("position_ids", position_ids.shape)
    self.llm_engine.set_runtime_tensor_shape("inputs_embeds", inputs_embeds.shape)
    self.llm_engine.set_runtime_tensor_shape("visual_pos_masks", visual_pos_masks.shape)
    self.llm_engine.set_runtime_tensor_shape("deepstack_visual_embeds", deepstack_visual_embeds.shape)
    hidden_states = self.llm_engine(position_ids, inputs_embeds, visual_pos_masks, deepstack_visual_embeds)["hidden_states"]
    return hidden_states


def vit_tensorrt_forward(self, hidden_states, grid_thw):
    print("vit_tensorrt_forward")
    print(f"{hidden_states.shape}, {hidden_states.dtype}, {hidden_states.device}")
    print(f"{grid_thw.shape}, {grid_thw.dtype}, {grid_thw.device}")

    self.vit_engine.set_runtime_tensor_shape("hidden_states", hidden_states.shape)
    self.vit_engine.set_runtime_tensor_shape("image_grid_thw", grid_thw.shape)
    vit_outputs = self.vit_engine(hidden_states, grid_thw) # "image_embeds", "deepstack_image_embeds"
    return vit_outputs


def vlm_tensorrt_forward(self, input_ids, attention_masks, image_embeds):
    print("vit_tensorrt_forward")
    print(f"{input_ids.shape}, {input_ids.dtype}, {input_ids.device}")
    print(f"{attention_masks.shape}, {attention_masks.dtype}, {attention_masks.device}")
    print(f"{image_embeds.shape}, {image_embeds.dtype}, {image_embeds.device}")


    self.vlm_engine.set_runtime_tensor_shape("input_ids", input_ids.shape)
    self.vlm_engine.set_runtime_tensor_shape("attention_masks", attention_masks.shape)
    self.vlm_engine.set_runtime_tensor_shape("image_embeds", image_embeds.shape)
    vlm_outputs = self.vlm_engine(input_ids, attention_masks, image_embeds)
    return vlm_outputs


def merge_tensorrt_forward(self, input_ids, attention_mask, pixel_values, image_grid_thw, **kwargs):
    # print("vit_tensorrt_forward")
    # print(f"{input_ids.shape}, {input_ids.dtype}, {input_ids.device}")
    # print(f"{attention_mask.shape}, {attention_mask.dtype}, {attention_mask.device}")
    # print(f"{pixel_values.shape}, {pixel_values.dtype}, {pixel_values.device}")
    # print(f"{image_grid_thw.shape}, {image_grid_thw.dtype}, {image_grid_thw.device}")

    self.vit_engine.set_runtime_tensor_shape("hidden_states", pixel_values.shape)
    self.vit_engine.set_runtime_tensor_shape("image_grid_thw", image_grid_thw.shape)
    vit_outputs = self.vit_engine(pixel_values, image_grid_thw)  # "image_embeds", "deepstack_image_embeds"

    self.vlm_engine.set_runtime_tensor_shape("input_ids", input_ids.shape)
    self.vlm_engine.set_runtime_tensor_shape("attention_masks", attention_mask.shape)
    self.vlm_engine.set_runtime_tensor_shape("image_embeds", vit_outputs["image_embeds"].shape)
    vlm_outputs = self.vlm_engine(input_ids, attention_mask, vit_outputs["image_embeds"])

    self.llm_engine.set_runtime_tensor_shape("position_ids", vlm_outputs["position_ids"].shape)
    self.llm_engine.set_runtime_tensor_shape("inputs_embeds", vlm_outputs["inputs_embeds"].shape)
    self.llm_engine.set_runtime_tensor_shape("visual_pos_masks", vlm_outputs["visual_pos_masks"].shape)
    self.llm_engine.set_runtime_tensor_shape("deepstack_visual_embeds", vit_outputs["deepstack_image_embeds"].shape)
    llm_output = self.llm_engine(vlm_outputs["position_ids"], vlm_outputs["inputs_embeds"],
                                    vlm_outputs["visual_pos_masks"], vit_outputs["deepstack_image_embeds"])

    self.gen_engine.set_runtime_tensor_shape("hidden_states", llm_output["hidden_states"].shape)
    gen_output = self.gen_engine(llm_output["hidden_states"])["logits"]

    return Qwen3VLCausalLMOutputWithPast(
            logits=gen_output
        )


def setup_tensorrt_engines(qwen_model, trt_engine_path):

    qwen_model.llm_engine = trt.Engine(os.path.join(trt_engine_path, "llm.engine"))
    qwen_model.vit_engine = trt.Engine(os.path.join(trt_engine_path, "vit.engine"))
    qwen_model.vlm_engine = trt.Engine(os.path.join(trt_engine_path, "vlm.engine"))
    qwen_model.gen_engine = trt.Engine(os.path.join(trt_engine_path, "gen.engine"))

    # qwen_model.model.forward = partial(llm_tensorrt_forward, qwen_model.model)
    # qwen_model.model.forward = partial(vit_tensorrt_forward, qwen_model.model)
    # qwen_model.model.forward = partial(vlm_tensorrt_forward, qwen_model.model)
    qwen_model.forward = partial(merge_tensorrt_forward, qwen_model)


def load_part_tensorrt(config):
    model_input = get_model_input(config)
    qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(config.qwen_path, dtype=torch.bfloat16,
                                                                 device_map='cpu', attn_implementation="eager")
    if hasattr(qwen_model.model, "language_model"):
        del qwen_model.model.language_model
    if hasattr(qwen_model.model, "vision_model"):
        del qwen_model.model.vision_model
    if hasattr(qwen_model, "lm_head"):
        del qwen_model.lm_head

    torch.cuda.empty_cache()
    setup_tensorrt_engines(qwen_model, os.path.join(config.onnx_path, "tensorrt"))

    # Create ONNX LLM inputs
    # input_ids = model_input["input_ids"]
    # batch_size, seq_len = input_ids.shape
    # deepstack_visual_len = 3
    #
    # position_ids = torch.ones((3, batch_size, seq_len), dtype=torch.int64).to(config.device)  # torch.Size([3, 1, 144])
    # inputs_embeds = torch.zeros((batch_size, seq_len, 2048),
    #                             dtype=torch.float16 if config.dtype == "fp16" else torch.float32).to(
    #     config.device)  # torch.Size([1, 144, 2048])
    #
    # visual_pos_masks = torch.rand(batch_size, seq_len) > 0.5
    # x = visual_pos_masks.sum().item()
    # visual_pos_masks = visual_pos_masks.to(config.device)  # torch.Size([1, 144])
    # deepstack_visual_embeds = torch.randn((deepstack_visual_len, x, 2048),
    #                                       dtype=torch.float16 if config.dtype == "fp16" else torch.float32).to(config.device)  # torch.Size([3, 67, 2048])

    # Test LLM
    # hidden_states = qwen_model.model(position_ids, inputs_embeds, visual_pos_masks, deepstack_visual_embeds)
    # print(f"{hidden_states.shape}, {hidden_states.dtype}, {hidden_states.device}")

    # Test VIT
    # pixel_values = model_input["pixel_values"].to(dtype=torch.float16 if config.dtype == "fp16" else torch.float32, device=config.device)
    # image_grid_thw = model_input["image_grid_thw"].to(dtype=torch.int64, device=config.device)
    # hidden_states = qwen_model.model(pixel_values, image_grid_thw)

    # Test VLM
    input_ids = model_input["input_ids"].clone().to(config.device)
    attention_masks = model_input["attention_mask"].clone().to(config.device)
    image_embeds = torch.randn((64, qwen_model.config.vision_config.out_hidden_size), dtype=torch.float16 if config.dtype=="fp16" else torch.float32).to(config.device)
    vlm_output = qwen_model.model(input_ids, attention_masks, image_embeds)
    # merge_output = qwen_model(input_ids, attention_masks, hidden_states, image_grid_thw)


def load_model_tensorrt(config):
    model_input = get_model_input(config)
    qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(config.qwen_path, dtype=torch.bfloat16,
                                                                 device_map='cpu', attn_implementation="eager")

    setup_tensorrt_engines(qwen_model, os.path.join(config.export_path, "tensorrt"))
    # Merge Model For Vision Language
    model_input["input_ids"] = model_input["input_ids"].to(dtype=torch.int64, device=config.device)
    model_input["attention_mask"] = model_input["attention_mask"].to(dtype=torch.int64, device=config.device)
    model_input["pixel_values"] = model_input["pixel_values"].to(dtype=torch.float16 if config.dtype=="fp16" else torch.float32, device=config.device)
    model_input["image_grid_thw"] = model_input["image_grid_thw"].to(dtype=torch.int64, device=config.device)
    with torch.no_grad():
        model_output = qwen_model.generate(**model_input, use_cache=False, max_length=1280)

    processor = AutoProcessor.from_pretrained(config.qwen_path)
    batch_outputs = processor.batch_decode(
        model_output,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    # 按批次顺序打印每个样本的输出
    for i, output in enumerate(batch_outputs):
        print(f"===== 样本 {i + 1} 输出 =====")
        print(output)
        print("\n" + "-" * 50 + "\n")

    start_time = time.perf_counter()
    for _ in range(1000):
        qwen_model.forward(**model_input)
    print(time.perf_counter() - start_time)


def compare_inference_speed(config):
    model_input = get_model_input(config)
    qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(config.qwen_path, dtype=torch.bfloat16,
                                                                 device_map='cpu', attn_implementation="eager")
    # qwen_start_time = time.perf_counter()
    # qwen_tokens = 0
    # for _ in range(10):
    #     with torch.no_grad():
    #         model_output = qwen_model.generate(**model_input, use_cache=False, max_length=1280)[:, model_input['input_ids'].shape[1]:]
    #         qwen_tokens += model_output.shape[0] * model_output.shape[1]
    # qwen_end_time = time.perf_counter()

    setup_tensorrt_engines(qwen_model, os.path.join(config.export_path, "tensorrt"))
    model_input["input_ids"] = model_input["input_ids"].to(dtype=torch.int64, device=config.device)
    model_input["attention_mask"] = model_input["attention_mask"].to(dtype=torch.int64, device=config.device)
    model_input["pixel_values"] = model_input["pixel_values"].to(dtype=torch.float16 if config.dtype=="fp16" else torch.float32, device=config.device)
    model_input["image_grid_thw"] = model_input["image_grid_thw"].to(dtype=torch.int64, device=config.device)

    tensorrt_start_time = time.perf_counter()
    tensorrt_tokens = 0
    for _ in range(10):
        with torch.no_grad():
            model_output = qwen_model.generate(**model_input, use_cache=False, max_length=1280)[:, model_input['input_ids'].shape[1]:]
            tensorrt_tokens += model_output.shape[0] * model_output.shape[1]
    tensorrt_end_time = time.perf_counter()
    # print(f"Qwen3-vl Generated tokens nums:{qwen_tokens}, speed: {(qwen_tokens / (qwen_end_time - qwen_start_time)): 2f} tokens/sec")
    print(f"Tensorrt Generated tokens nums:{tensorrt_tokens}, speed: {(tensorrt_tokens / (tensorrt_end_time - tensorrt_start_time)): 2f} tokens/sec")




if __name__ == "__main__":
    device = 'cuda'
    config = ArgsConfig()
    config.device = device
    config.export_path += "_" + config.dtype
    # load_model_tensorrt(config)
    compare_inference_speed(config)