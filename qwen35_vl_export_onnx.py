import os
import shutil
import torch
from transformers import Qwen3_5ForConditionalGeneration, Qwen3VLForConditionalGeneration
from modules import Qwen3VLTextModelOpt, Qwen3VLVisualModelOpt, Qwen3VLModelOpt, Qwen3VLForConditionalGenerationOpt
from utils import get_model_input, get_qwen3_onnx_input
from config.qwen35_config import ArgsConfig


def export_part_onnx(qwen_model, opt_model, onnx_inputs, onnx_path, config):

    if config.dtype == torch.float16:
        opt_model.half()

    opt_model.load_state_dict(qwen_model.state_dict())
    opt_model = opt_model.to(config.device)

    torch.onnx.export(
        opt_model,
        onnx_inputs["inputs"],
        onnx_path,
        input_names=onnx_inputs["input_names"],
        output_names=onnx_inputs["output_names"],
        dynamic_axes=onnx_inputs["dynamic_axes"],
        verbose=True,
    )



def run_export(config: ArgsConfig):
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.dtype = torch.float16 if config.dtype=="fp16" else torch.float32
    torch.manual_seed(42)

    model_input = get_model_input(config)
    qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(config.qwen_path, dtype=torch.float32, device_map='cpu', attn_implementation="eager")
    print("Init model load done!")
    onnx_input = get_qwen3_onnx_input(
        config,
        model_input,
        llm_hidden_size=qwen_model.model.language_model.config.hidden_size,
        vit_hidden_size=qwen_model.model.config.vision_config.out_hidden_size,
        gen_hidden_size=qwen_model.config.text_config.hidden_size,
    )

    print("Export ONNX model type: ", config.dtype)
    onnx_path = config.export_path + "/ONNX"

    part_qwen_model = {
        "llm": {
            "original": qwen_model.model.language_model,
            "optimized": Qwen3VLTextModelOpt(qwen_model.model.language_model.config),
        },
        "vit": {
            "original": qwen_model.model.visual,
            "optimized": Qwen3VLVisualModelOpt(qwen_model.model.visual.config),
        },
        "vlm": {
            "original": qwen_model.model,
            "optimized": Qwen3VLModelOpt(qwen_model.model.config, config),
        },
        "gen": {
            "original": qwen_model,
            "optimized": Qwen3VLForConditionalGenerationOpt(qwen_model.config),
        },
    }


    for part_model_name in onnx_input.keys():
        # remove the previous dir
        onnx_part_dir = os.path.join(onnx_path, part_model_name)
        if os.path.exists(onnx_part_dir):
            shutil.rmtree(onnx_part_dir)
        os.makedirs(onnx_part_dir)

        export_part_onnx(
            qwen_model=part_qwen_model[part_model_name]["original"],
            opt_model=part_qwen_model[part_model_name]["optimized"],
            onnx_inputs=onnx_input[part_model_name],
            onnx_path=os.path.join(onnx_path, part_model_name, part_model_name + ".onnx"),
            config=config,
        )

    print("Export Qwen3 done!")


if __name__ == "__main__":
    cfg = ArgsConfig()
    run_export(cfg)