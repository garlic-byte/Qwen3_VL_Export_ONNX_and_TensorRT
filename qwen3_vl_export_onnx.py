import os
import shutil
import torch
from transformers import Qwen3VLForConditionalGeneration
from dataclasses import dataclass, field
from modules import Qwen3VLTextModelOpt, Qwen3VLVisualModelOpt, Qwen3VLModelOpt, Qwen3VLForConditionalGenerationOpt
from utils import get_model_input


@dataclass
class ArgsConfig:
    """Configuration for Qwen3-VL model export ONNX"""

    # Model parameters
    qwen_path: str = '/home/wsj/Desktop/data/Downloads/weights/qwen3-vl-4b'
    """Path to the qwen directory or directories"""

    export_path: str = 'qwen3_vl_4b'
    """Directory to save onnx model checkpoints."""

    batch_size: int = 1
    """Batch size of input for ONNX model inference"""

    imgs_nums: int = 1
    """Number of images for ONNX model inference"""

    dtype: str = 'fp16'
    """Data type of ONNX model: 'fp16' or 'fp32' """

    device: str = None
    """Device used for ONNX model inference"""


def export_qwen_llm(qwen_model, inputs, onnx_path, config):
    # Remove and create new onnx dir
    dir_path = os.path.dirname(onnx_path)
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

    # qwen_model.config._attn_implementation = "eager"
    # Create Text Model
    model = Qwen3VLTextModelOpt(qwen_model.config)
    if config.dtype == "fp16":
        model.half()
    model = model.to(config.device)
    model.load_state_dict(qwen_model.state_dict())
    model.eval()

    # Create ONNX inputs
    input_ids = inputs["input_ids"]
    batch_size, seq_len = input_ids.shape
    deepstack_visual_len = 3

    position_ids = torch.ones((3, batch_size, seq_len), dtype=torch.int64).to(config.device) # torch.Size([3, 1, 144])
    inputs_embeds = torch.zeros((batch_size, seq_len, qwen_model.config.hidden_size), dtype=torch.float16 if config.dtype == "fp16" else torch.float32).to(config.device) # torch.Size([1, 144, 2048])

    visual_pos_masks = torch.rand(batch_size, seq_len) > 0.5
    x = visual_pos_masks.sum().item()
    visual_pos_masks = visual_pos_masks.to(config.device) # torch.Size([1, 144])
    deepstack_visual_embeds = torch.randn((deepstack_visual_len, x, qwen_model.config.hidden_size), dtype=torch.float16 if config.dtype == "fp16" else torch.float32).to(config.device) # torch.Size([3, 67, 2048])

    torch.onnx.export(
        model,
        (position_ids, inputs_embeds, visual_pos_masks, deepstack_visual_embeds),
        onnx_path,
        export_params=True,
        do_constant_folding=True,
        input_names=["position_ids", "inputs_embeds", "visual_pos_masks", "deepstack_visual_embeds"],
        output_names=["hidden_states"],
        dynamic_axes={
            "position_ids": {1: "batch_size", 2: "seq_length"},
            "inputs_embeds": {0: "batch_size", 1: "seq_length"},
            "visual_pos_masks": {0: "batch_size", 1: "seq_length"},
            "deepstack_visual_embeds": {1: "visual_seqlen"},
            "hidden_states": {0: "batch_size", 1: "seq_length"},
        },
        verbose=True,
    )

    print("Export Qwen3 LLM done!")
    del position_ids, inputs_embeds, visual_pos_masks, deepstack_visual_embeds
    del model

def export_qwen_vit(qwen_model, inputs, onnx_path, config):

    dir_path = os.path.dirname(onnx_path)
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

    model = Qwen3VLVisualModelOpt(qwen_model.config).to(config.device)
    if config.dtype == "fp16":
        model.half()
    model.load_state_dict(qwen_model.state_dict())
    model = model.to(config.device)
    model.eval()

    hidden_states = inputs["pixel_values"].clone() # seq_len x 1536
    image_grid_thw = inputs["image_grid_thw"].clone() # img_num x 3
    if config.dtype == "fp16":
        hidden_states = hidden_states.to(dtype=torch.float16)

    torch.onnx.export(
        model,
        (hidden_states, image_grid_thw),
        onnx_path,
        input_names=["hidden_states", "image_grid_thw"],
        output_names=["image_embeds", "deepstack_image_embeds"],
        dynamic_axes={
            "hidden_states": {0: "seq_len"},
            "image_grid_thw": {0: "img_num"},
        },
        verbose=True,
    )

    print("Export Qwen3 Vit done!")
    del hidden_states, image_grid_thw
    del model

def export_qwen_vlm(qwen_model, inputs, onnx_path, config):

    dir_path = os.path.dirname(onnx_path)
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

    model = Qwen3VLModelOpt(qwen_model.config, config)
    if config.dtype == "fp16":
        model.half()
    model.load_state_dict(qwen_model.state_dict())
    model = model.to(config.device)
    model.eval()

    input_ids = inputs["input_ids"].clone()    # shape torch.Size([1, 144])
    attention_masks = inputs["attention_mask"].clone()    # shape torch.Size([1, 144])
    pixel_values = inputs["pixel_values"].clone().to(dtype=torch.float16 if config.dtype=="fp16" else torch.float32)    # shape torch.Size([512, 1536])
    image_grid_thw = inputs["image_grid_thw"].clone()    # shape torch.Size([2, 3])
    image_embeds = torch.randn((64, qwen_model.config.vision_config.out_hidden_size), dtype=torch.float16 if config.dtype=="fp16" else torch.float32)
    deepstack_image_embeds = torch.randn((3, 64, qwen_model.config.vision_config.out_hidden_size), dtype=torch.float16 if config.dtype=="fp16" else torch.float32)

    torch.onnx.export(
        model,
        (input_ids, attention_masks, image_embeds),
        onnx_path,
        input_names=["input_ids", "attention_masks", "image_embeds"],
        output_names=["position_ids", "attention_mask", "inputs_embeds", "visual_pos_masks"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_length"},
            "attention_masks": {0: "batch_size", 1: "seq_length"},
            "image_grid_thw": {0: "num_images"},
            # "image_embeds": {0: "hidden_dim"},
        },
        verbose=True,
    )

    print("Export Qwen3 Vit done!")
    del input_ids, attention_masks, pixel_values, image_grid_thw
    del model

def export_qwen_gen(qwen_model, inputs, onnx_path, config):
    dir_path = os.path.dirname(onnx_path)
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

    model = Qwen3VLForConditionalGenerationOpt(qwen_model.config)
    if config.dtype == "fp16":
        model.half()

    model.load_state_dict(qwen_model.state_dict())
    model = model.to(config.device)

    input_ids = inputs["input_ids"]
    batch_size, seq_len = input_ids.shape
    hidden_states = torch.randn((batch_size, seq_len, qwen_model.config.text_config.hidden_size), dtype=torch.float16 if config.dtype=="fp16" else torch.float32).to(config.device)
    torch.onnx.export(
        model,
        (hidden_states,),
        onnx_path,
        input_names=["hidden_states"],
        output_names=["logits"],
        dynamic_axes={
            "hidden_states": {0: "batch_size", 1: "seq_length"},
        },
        verbose=True,
    )

    print("Export Qwen3 Generate done!")
    del hidden_states
    del model


def run_export(config: ArgsConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.device = device
    torch.manual_seed(42)

    model_input = get_model_input(config)
    qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(config.qwen_path, dtype=torch.float32, device_map='cpu', attn_implementation="eager")
    print("Init model load done!")

    print("Export ONNX model type: ", config.dtype)
    config.onnx_path = config.export_path + "_" + config.dtype + "/ONNX"
    export_qwen_llm(
        qwen_model=qwen_model.model.language_model,
        inputs=model_input,
        onnx_path=os.path.join(config.onnx_path, 'llm/llm.onnx'),
        config=config
    )
    export_qwen_vit(
        qwen_model=qwen_model.model.visual,
        inputs=model_input,
        onnx_path=os.path.join(config.onnx_path, 'vit/vit.onnx'),
        config=config,
    )
    export_qwen_vlm(
        qwen_model=qwen_model.model,
        inputs=model_input,
        onnx_path=os.path.join(config.onnx_path, 'vlm/vlm.onnx'),
        config=config,
    )
    export_qwen_gen(
        qwen_model=qwen_model,
        inputs=model_input,
        onnx_path=os.path.join(config.onnx_path, 'gen/gen.onnx'),
        config=config,
    )


if __name__ == "__main__":
    cfg = ArgsConfig()
    run_export(cfg)