from dragon_export_onnx import get_model_input, qwen_path, COMPUTE_DTYPE, onnx_qwen_vit, onnx_qwen_llm, device
import torch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLVisionModel, AutoTokenizer
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding, Qwen3VLModel, Qwen3VLTextModel, create_causal_mask
import onnx
import onnxruntime as ort
import netron
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


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

def load_onnx_llm(model_input, trt_engine_path):


    onnx_inputs = {
        # 从 GPU 迁移到 CPU，再转为 NumPy，保持原数据类型
        "position_ids": model_input["position_ids"].cpu().numpy(),  # torch.int64 → np.int64
        "inputs_embeds": model_input["inputs_embeds"].cpu().numpy(),  # torch.float32 → np.float32
        "visual_pos_masks": model_input["visual_pos_masks"].cpu().numpy(),  # torch.bool → np.bool_
        "deepstack_visual_embeds": model_input["deepstack_visual_embeds"].cpu().numpy()  # torch.float32 → np.float32
    }

    check_info(trt_engine_path, onnx_inputs)

    # 执行推理（output_names 为要获取的输出，若为 None 则返回所有输出）
    session = ort.InferenceSession(
        trt_engine_path,
        providers=["CUDAExecutionProvider"]
    )


    input_names = [input.name for input in session.get_inputs()]
    output_names = [output.name for output in session.get_outputs()]
    print("ONNX模型输入名称:", input_names)
    print("ONNX模型输出名称:", output_names)

    onnx_outputs = session.run(
        input_feed=onnx_inputs,
        output_names=["hidden_states"],
    )

    return onnx_outputs[0]


def load_onnx_vit(model_input, trt_engine_path):

    pixel_values = model_input["pixel_values"]
    image_grid_thw = model_input["image_grid_thw"]


    onnx_inputs = {
        # 从 GPU 迁移到 CPU，再转为 NumPy，保持原数据类型
        "pixel_values": pixel_values.cpu().numpy(),  # torch.int64 → np.int64
        "image_grid_thw": image_grid_thw.cpu().numpy(),  # torch.float32 → np.float32
    }

    #
    check_info(trt_engine_path, onnx_inputs)


    # 执行推理（output_names 为要获取的输出，若为 None 则返回所有输出）
    session = ort.InferenceSession(
        trt_engine_path,
        providers=["CUDAExecutionProvider"]
    )


    input_names = [input.name for input in session.get_inputs()]
    output_names = [output.name for output in session.get_outputs()]
    print("ONNX模型输入名称:", input_names)
    print("ONNX模型输出名称:", output_names)

    onnx_outputs = session.run(
        input_feed=onnx_inputs,
        output_names=['hidden_states', "deepstack_feature_0", "deepstack_feature_1", "deepstack_feature_2"],
    )
    # onnx_hidden = onnx_outputs

    return onnx_outputs


def compare_vit(model_input, onnx_qwen_vit, torch_qwen_vit):

    onnx_outputs = load_onnx_vit(model_input, onnx_qwen_vit)

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


        from dragon_export_onnx import Qwen3VLTextModelOpt
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



def load_model_onnx():
    model_input = get_model_input(device)
    qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(qwen_path, dtype=COMPUTE_DTYPE, device_map='cpu', attn_implementation="eager")
    # netron.start(onnx_qwen_llm)

    # load_onnx_vit(model_input, onnx_qwen_vit)
    # load_onnx_llm(model_input, onnx_qwen_llm)
    # compare_vit(model_input, onnx_qwen_vit, qwen_model.model.visual)
    compare_llm(onnx_qwen_llm, qwen_model.model.language_model)
    # compare_llm_model(qwen_model.model.language_model)

if __name__ == "__main__":
    load_model_onnx()