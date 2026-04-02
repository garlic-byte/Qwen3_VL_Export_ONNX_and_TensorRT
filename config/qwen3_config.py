from dataclasses import dataclass


@dataclass
class ArgsConfig:
    """Configuration for Qwen3-VL model export ONNX"""
    # Model parameters
    qwen_path: str = '/home/wsj/Desktop/data/Downloads/weights/qwen3-vl-2b'
    """Path to the qwen directory or directories"""

    export_path: str = 'export/qwen3_vl_2b'
    """Directory to save onnx model checkpoints."""

    batch_size: int = 1
    """Batch size of input for ONNX model inference"""

    imgs_paths: tuple = ("demo_data/demo.jpeg", )
    """Path of images for ONNX model inference"""

    device: str = None
    """Device used for ONNX model inference"""

    dtype = 'fp16'
    """Data type of ONNX model: 'fp16' or 'fp32' """