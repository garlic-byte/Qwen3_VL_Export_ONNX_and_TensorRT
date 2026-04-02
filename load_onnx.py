import os
import time

from qwen3_vl_export_onnx import Qwen3VLVisualModelOpt
import torch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLVisionModel, AutoTokenizer, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding, Qwen3VLModel, Qwen3VLTextModel, create_causal_mask
import onnx
import onnxruntime as ort
import netron
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from utils import get_model_input, get_qwen35_onnx_input
from config.qwen35_config import ArgsConfig


def compare_predictions(pred_tensorrt, pred_torch) -> None:
    """
    Compare the similarity between TensorRT and PyTorch predictions

    Args:
        pred_tensorrt: TensorRT prediction results (numpy array)
        pred_torch: PyTorch prediction results (numpy array)
    """
    print("\n=== Prediction Comparison ===")

    # Ensure both predictions contain the same keys
    assert len(pred_tensorrt) == len(pred_torch), "Prediction length do not match"

    # Calculate max label width for alignment
    max_label_width = max(
        len("Cosine Similarity (PyTorch/TensorRT):"),
        len("L1 Mean/Max Distance (PyTorch/TensorRT):"),
        len("Max Output Values (PyTorch/TensorRT):"),
        len("Mean Output Values (PyTorch/TensorRT):"),
        len("Min Output Values (PyTorch/TensorRT):"),
    )

    for tensorrt_array, torch_array in zip(pred_tensorrt, pred_torch):

        # Convert to PyTorch tensors
        tensorrt_tensor = torch.from_numpy(tensorrt_array).to(torch.float32)
        torch_tensor = torch.from_numpy(torch_array).to(torch.float32)

        # Ensure tensor shapes are the same
        assert (
            tensorrt_tensor.shape == torch_tensor.shape
        ), f"Shapes do not match: {tensorrt_tensor.shape} vs {torch_tensor.shape}"

        # Calculate cosine similarity
        flat_tensorrt = tensorrt_tensor.flatten()
        flat_torch = torch_tensor.flatten()

        # Manually calculate cosine similarity
        dot_product = torch.dot(flat_tensorrt, flat_torch)
        norm_tensorrt = torch.norm(flat_tensorrt)
        norm_torch = torch.norm(flat_torch)
        cos_sim = dot_product / (norm_tensorrt * norm_torch)

        # Calculate L1 distance
        l1_dist = torch.abs(flat_tensorrt - flat_torch)

        print(f"\n:")
        print(f'{"Cosine Similarity (PyTorch/TensorRT):".ljust(max_label_width)} {cos_sim.item()}')
        print(
            f'{"L1 Mean/Max Distance (PyTorch/TensorRT):".ljust(max_label_width)} {l1_dist.mean().item():.4f}/{l1_dist.max().item():.4f}'
        )
        print(
            f'{"Max Output Values (PyTorch/TensorRT):".ljust(max_label_width)} {torch_tensor.max().item():.4f}/{tensorrt_tensor.max().item():.4f}'
        )
        print(
            f'{"Mean Output Values (PyTorch/TensorRT):".ljust(max_label_width)} {torch_tensor.mean().item():.4f}/{tensorrt_tensor.mean().item():.4f}'
        )
        print(
            f'{"Min Output Values (PyTorch/TensorRT):".ljust(max_label_width)} {torch_tensor.min().item():.4f}/{tensorrt_tensor.min().item():.4f}'
        )



def check_info(onnx_path, onnx_inputs):
    # 加载 ONNX 模型
    onnx_model = onnx.load(onnx_path)

    inputs_info = {}
    # 获取输入节点信息
    print("=== ONNX Model Input dim ===")
    for input_node in onnx_model.graph.input:
        dims = [dim.dim_value if dim.dim_value != 0 else "dynamic" for dim in input_node.type.tensor_type.shape.dim]
        inputs_info[input_node.name] = tuple(dims)
        print(f"{input_node.name}: {dims}")

    # 获取输出节点信息
    print("\n=== ONNX Model Output dim ===")
    for output_node in onnx_model.graph.output:
        dims = [dim.dim_value if dim.dim_value != 0 else "dynamic" for dim in output_node.type.tensor_type.shape.dim]
        print(f"{output_node.name}: {dims}")


    # 4. 验证转换后的类型和形状（可选，用于调试）
    print("\n=== Ready for ONNX inputs ===")
    for name, arr in onnx_inputs.items():
        print(f"{name}: 形状={arr.shape}, 类型={arr.dtype}")
        # assert inputs_info[name] == arr.shape, f"Need input {name} shape == {inputs_info[name]} but got {arr.shape} !!!"


def load_onnx_part(inputs, onnx_part):
    onnx_inputs = {}
    for input_name, input_tensor in zip(inputs["input_names"], inputs["inputs"]):
        onnx_inputs[input_name] = input_tensor.cpu().numpy()

    check_info(onnx_part, onnx_inputs)

    # 执行推理（output_names 为要获取的输出，若为 None 则返回所有输出）
    session = ort.InferenceSession(
        onnx_part,
        providers=["CUDAExecutionProvider"]
    )


    input_names = [input.name for input in session.get_inputs()]
    output_names = [output.name for output in session.get_outputs()]
    print("ONNX模型输入名称:", input_names)
    print("ONNX模型输出名称:", output_names)

    onnx_outputs = session.run(
        input_feed=onnx_inputs,
        output_names=inputs["output_names"],
    )

    return onnx_outputs[0]


def load_onnx_vit(inputs, onnx_vit):

    hidden_states = inputs["pixel_values"].clone() # seq_len x 1536
    image_grid_thw = inputs["image_grid_thw"].clone() # img_num x 3
    if config.dtype == "fp16":
        hidden_states = hidden_states.to(dtype=torch.float16)

    onnx_inputs = {
        "hidden_states": hidden_states.cpu().numpy(),
        "image_grid_thw": image_grid_thw.cpu().numpy(),
    }

    check_info(onnx_vit, onnx_inputs)

    session = ort.InferenceSession(
        onnx_vit,
        providers=["CUDAExecutionProvider"]
    )


    input_names = [input.name for input in session.get_inputs()]
    output_names = [output.name for output in session.get_outputs()]
    print("ONNX inputs name:", input_names)
    print("ONNX outputs name:", output_names)

    onnx_outputs = session.run(
        input_feed=onnx_inputs,
        output_names=["image_embeds", "deepstack_image_embeds"],
    )
    # print(onnx_outputs)
    return onnx_outputs


def load_onnx_vlm(inputs, onnx_vlm):

    input_ids = inputs["input_ids"].clone()    # shape torch.Size([1, 144])
    attention_masks = inputs["attention_mask"].clone()    # shape torch.Size([1, 144])
    pixel_values = inputs["pixel_values"].clone()    # shape torch.Size([512, 1536])
    image_grid_thw = inputs["image_grid_thw"].clone()    # shape torch.Size([2, 3])
    image_embeds = torch.randn((64, 2048), dtype=torch.float16 if config.dtype == "fp16" else torch.float32)
    deepstack_image_embeds = torch.randn((3, 64, 2048),
                                           dtype=torch.float16 if config.dtype == "fp16" else torch.float32)

    if config.dtype == "fp16":
        pixel_values = pixel_values.to(dtype=torch.float16)



    onnx_inputs = {
        "input_ids": input_ids.cpu().numpy(),
        "attention_masks": attention_masks.cpu().numpy(),
        "image_embeds": image_embeds.cpu().numpy(),
    }

    check_info(onnx_vlm, onnx_inputs)

    session = ort.InferenceSession(
        onnx_vlm,
        providers=["CUDAExecutionProvider"]
    )


    input_names = [input.name for input in session.get_inputs()]
    output_names = [output.name for output in session.get_outputs()]
    print("ONNX inputs name:", input_names)
    print("ONNX outputs name:", output_names)

    onnx_outputs = session.run(
        input_feed=onnx_inputs,
        output_names=["position_ids", "attention_mask", "inputs_embeds", "visual_pos_masks"],
    )
    # print(onnx_outputs)
    return onnx_outputs


def load_onnx_gen(inputs, onnx_gen):
    input_ids = inputs["input_ids"].clone()
    batch_size, hidden_size = input_ids.shape
    hidden_states = torch.randn((batch_size, hidden_size, 2048), dtype=torch.float16 if config.dtype == "fp16" else torch.float32)

    # VLM process
    gen_inputs = {
        "hidden_states": hidden_states.cpu().numpy(),
    }

    gen_session = ort.InferenceSession(
        onnx_gen,
        providers=["CUDAExecutionProvider"]
    )
    check_info(onnx_gen, gen_inputs)

    gen_outputs = gen_session.run(
        input_feed=gen_inputs,
        output_names=["logits"],
    )

    return gen_outputs[0]


def compare_vit(model_input, onnx_qwen_vit, torch_qwen_vit):

    onnx_outputs = load_onnx_vlm(model_input, onnx_qwen_vit)

    torch_qwen_vit = torch_qwen_vit.to(device)
    with torch.no_grad():
        torch_outputs_tuple = torch_qwen_vit(
            hidden_states=model_input["pixel_values"],
            grid_thw=model_input["image_grid_thw"],
        )

    torch_outputs_list = [torch_outputs_tuple[0]] + torch_outputs_tuple[1]
    torch_outputs = [l.cpu().numpy() for l in torch_outputs_list]

    compare_predictions(onnx_outputs, torch_outputs)


def compare_llm(onnx_qwen_llm, torch_qwen_llm):
    torch_qwen_llm = torch_qwen_llm.to(device)

    batch_size = 1
    seq_len = 144
    hidden_size = 2048
    deepstack_visual_len = 3
    torch.manual_seed(42)
    visual_pos_masks = torch.rand(batch_size, seq_len) > 0.5
    x = visual_pos_masks.sum().item()

    base_pos = torch.arange(seq_len)
    position_ids = base_pos.view(1, 1, seq_len).repeat(3, 1, 1).to(device)
    inputs_embeds = torch.randn((batch_size, seq_len, hidden_size), dtype=torch.float32).to(device)
    visual_pos_masks = visual_pos_masks.to(device)
    deepstack_visual_embeds = torch.randn((deepstack_visual_len, x, 2048), dtype=torch.float32).to(device)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.int64).to(device)

    model_input = {}
    model_input["position_ids"] = position_ids.clone()
    model_input["inputs_embeds"] = inputs_embeds.clone()
    model_input["visual_pos_masks"] = visual_pos_masks.clone()
    model_input["deepstack_visual_embeds"] = deepstack_visual_embeds.clone()

    onnx_outputs = load_onnx_llm(model_input, onnx_qwen_llm)

    # torch_qwen_llm = torch_qwen_llm.to(device)
    torch_qwen_llm.config._attn_implementation = "eager"
    with torch.no_grad():
        torch_outputs = torch_qwen_llm(
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )["last_hidden_state"]

    compare_predictions([onnx_outputs], [torch_outputs.cpu().numpy()])


def compare_llm_model(torch_qwen_llm):
    # torch_qwen_llm.config._attn_implementation = "eager"
    torch_qwen_llm = torch_qwen_llm.to(device)
    batch_size = 1
    seq_len = 144
    hidden_size = 2048
    deepstack_visual_len = 3
    torch.manual_seed(42)
    visual_pos_masks = torch.rand(batch_size, seq_len) > 0.5
    x = visual_pos_masks.sum().item()

    base_pos = torch.arange(seq_len)
    position_ids = base_pos.view(1, 1, seq_len).repeat(3, 1, 1).to(device)
    inputs_embeds = torch.randn((batch_size, seq_len, hidden_size), dtype=torch.float32).to(device)
    visual_pos_masks = visual_pos_masks.to(device)
    deepstack_visual_embeds = torch.randn((deepstack_visual_len, x, 2048), dtype=torch.float32).to(device)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.int64).to(device)

    origin_outputs = []
    new_outputs = []

    with torch.no_grad():
        origin_outputs.append(
            torch_qwen_llm(
            position_ids=position_ids.clone(),
            inputs_embeds=inputs_embeds.clone(),
            visual_pos_masks=visual_pos_masks.clone(),
            deepstack_visual_embeds=deepstack_visual_embeds.clone(),
            )["last_hidden_state"].cpu().numpy()
        )


        from qwen3_vl_export_onnx import Qwen3VLTextModelOpt
        # torch_qwen_llm.config._attn_implementation = "eager"
        refactor_model = Qwen3VLTextModelOpt(torch_qwen_llm.config)
        refactor_model.load_state_dict(torch_qwen_llm.state_dict())
        refactor_model = refactor_model.to(device)
        new_outputs.append(
            refactor_model(
            position_ids=position_ids.clone(),
            inputs_embeds=inputs_embeds.clone(),
            visual_pos_masks=visual_pos_masks.clone(),
            deepstack_visual_embeds=deepstack_visual_embeds,
            ).cpu().numpy()
        )

    print(f"{torch_qwen_llm.config._attn_implementation} vs {refactor_model.config._attn_implementation}")
    compare_predictions(origin_outputs, new_outputs)


def compare_vit_model(torch_vit, input):
    torch_vit = torch_vit.to(device)
    model = Qwen3VLVisualModelOpt(torch_vit.config).to(device)
    model.load_state_dict(torch_vit.state_dict())
    model = model.to(device)

    input_ids = input["input_ids"].clone()    # shape torch.Size([1, 144])
    attention_mask = input["attention_mask"].clone()    # shape torch.Size([1, 144])
    pixel_values = input["pixel_values"].clone()    # shape torch.Size([512, 1536])
    image_grid_thw = input["image_grid_thw"].clone()    # shape torch.Size([2, 3])


    with torch.no_grad():
        origin_outputs = torch_vit(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )["last_hidden_state"].cpu().numpy()

        new_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )["last_hidden_state"].cpu().numpy()

        compare_predictions([origin_outputs], [new_outputs])


def compare_vlm(inputs, onnx_vit, onnx_llm, onnx_vlm, torch_vlm):
    input_ids = inputs["input_ids"].clone()    # shape torch.Size([1, 144])
    attention_masks = inputs["attention_mask"].clone()    # shape torch.Size([1, 144])
    pixel_values = inputs["pixel_values"].clone()    # shape torch.Size([512, 1536])
    image_grid_thw = inputs["image_grid_thw"].clone()    # shape torch.Size([2, 3])

    with torch.no_grad():
        orign_outputs = torch_vlm(
            input_ids=input_ids,
            attention_mask=attention_masks,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            logits_to_keep=1,
        )["logits"].cpu().numpy().squeeze(0)

        new_outputs = load_onnx_gen(inputs, onnx_vit, onnx_llm, onnx_vlm)

        compare_predictions([orign_outputs], [new_outputs])


def compare_vlm_speed(inputs, config):
    input_ids = inputs["input_ids"].clone()    # shape torch.Size([1, 144])
    attention_masks = inputs["attention_mask"].clone()    # shape torch.Size([1, 144])
    pixel_values = inputs["pixel_values"].clone().to(dtype=torch.float16 if config.dtype=='fp16' else torch.float32)    # shape torch.Size([512, 1536])
    image_grid_thw = inputs["image_grid_thw"].clone()    # shape torch.Size([2, 3])

    torch_vlm = Qwen3VLForConditionalGeneration.from_pretrained(config.qwen_path, dtype=torch.float32, device_map='cuda', attn_implementation="eager")
    with torch.no_grad():
        start_time = time.perf_counter()
        for _ in range(1000):
            orign_outputs = torch_vlm(
                input_ids=input_ids,
                attention_mask=attention_masks,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                logits_to_keep=1,
            )
        torch_spend = time.perf_counter() - start_time
    del torch_vlm, orign_outputs
    torch.cuda.empty_cache()

    vit_session = ort.InferenceSession(
        os.path.join(config.onnx_path, 'vit/vit.onnx'),
        providers=["CUDAExecutionProvider"]
    )
    llm_session = ort.InferenceSession(
        os.path.join(config.onnx_path, 'llm/llm.onnx'),
        providers=["CUDAExecutionProvider"]
    )
    vlm_session = ort.InferenceSession(
        os.path.join(config.onnx_path, 'vlm/vlm.onnx'),
        providers=["CUDAExecutionProvider"]
    )

    inputs = {
        "input_ids": input_ids.cpu().numpy(),
        "attention_masks": attention_masks.cpu().numpy(),
        "pixel_values": pixel_values.cpu().numpy(),  # torch.int64 → np.int64
        "image_grid_thw": image_grid_thw.cpu().numpy(),  # torch.float32 → np.float32
    }

    start_time = time.perf_counter()
    for _ in range(1000):
        vit_outputs = vit_session.run(
            input_feed=inputs,
            output_names=["position_ids", "attention_mask", "inputs_embeds", "visual_pos_masks",
                          "deepstack_visual_embeds"],
        )

        # LLM process
        llm_inputs = {
            "position_ids": vit_outputs[0],
            "inputs_embeds": vit_outputs[2],
            "visual_pos_masks": vit_outputs[3],
            "deepstack_visual_embeds": vit_outputs[4]
        }
        llm_outputs = llm_session.run(
            input_feed=llm_inputs,
            output_names=["hidden_states"],
        )

        # VLM process
        vlm_inputs = {
            "hidden_states": llm_outputs[0],
        }
        vlm_outputs = vlm_session.run(
            input_feed=vlm_inputs,
            output_names=["logits"],
        )
    onnx_spend = time.perf_counter() - start_time

    print("Torch speed time: ", torch_spend)
    print("ONNX speed time: ", onnx_spend)



def load_model_onnx(config):
    model_input = get_model_input(config)
    onnx_path = os.path.join(config.export_path, 'ONNX')
    qwen_model = Qwen3_5ForConditionalGeneration.from_pretrained(config.qwen_path, dtype=torch.float32,
                                                                 device_map='cpu', attn_implementation="eager")

    onnx_input = get_qwen35_onnx_input(
        config,
        model_input,
        llm_hidden_size=qwen_model.model.language_model.config.hidden_size,
        vit_hidden_size=qwen_model.model.config.vision_config.out_hidden_size,
        gen_hidden_size=qwen_model.config.text_config.hidden_size,
    )
    # with torch.no_grad():
    #     model_output = qwen_model.generate(**model_input, use_cache=False)
    #     print(model_output)
        # model_output = qwen_model(**model_input, output_hidden_states=True, use_cache=False, return_dict=True)

    # export_onnx_file = os.path.join(config.onnx_path, 'vit/vit.onnx')
    # simpler_onnx(os.path.join(config.onnx_path, 'vit/vit.onnx'))


    print("ONNX model load done!")

    part_name = 'vlm'
    netron.start(os.path.join(onnx_path, f'{part_name}/{part_name}.onnx'))

    load_onnx_part(
        inputs=onnx_input[part_name],
        onnx_part=os.path.join(onnx_path, f'{part_name}/{part_name}.onnx')
    )
    # compare_vit(model_input, onnx_qwen_vit, qwen_model.model.visual)
    # compare_llm(onnx_qwen_llm, qwen_model.model.language_model)
    # compare_llm_model(qwen_model.model.language_model)
    # compare_vit_model(qwen_model.model, model_input)
    # load_onnx_vit(
    #     inputs=model_input,
    #     onnx_vit=os.path.join(config.onnx_path, 'vit/vit.onnx')
    # )
    # load_onnx_vlm(
    #     inputs=model_input,
    #     onnx_vlm=os.path.join(onnx_path, 'vlm/vlm.onnx')
    # )
    # load_onnx_gen(
    #     inputs=model_input,
    #     onnx_gen=os.path.join(onnx_path, 'gen/gen.onnx'),
    # )
    # compare_vlm(
    #     inputs=model_input,
    #     onnx_vit=os.path.join(config.onnx_path, 'vit/vit.onnx'),
    #     onnx_llm=os.path.join(config.onnx_path, 'llm/llm.onnx'),
    #     onnx_vlm=os.path.join(config.onnx_path, 'vlm/vlm.onnx'),
    #     torch_vlm=qwen_model
    # )
    # compare_vlm_speed(
    #     inputs=model_input,
    #     config=config
    # )


if __name__ == "__main__":
    device = 'cuda'
    cfg = ArgsConfig()
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.dtype = torch.float16 if cfg.dtype=="fp16" else torch.float32
    load_model_onnx(cfg)