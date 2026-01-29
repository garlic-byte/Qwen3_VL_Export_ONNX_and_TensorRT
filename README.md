# qwen3-vl-2b-ONNX: Image-to-Text Inference Model

## Overview
This repository provides the ONNX-converted version of the **qwen3-vl-2b** multimodal model, optimized for efficient image-to-text generation. The model supports inference on single images with a fixed input resolution of 224×224 and outputs descriptive text based on visual content.

## Key Features
- **Model Type**: ONNX-exported multimodal large language model (vision-language)
- **Input Specification**: Single RGB image (224×224 resolution, 3 channels)
- **Output**: High-quality natural language description of the input image
- **Conversion Source**: Original qwen3-vl-2b (PyTorch) → ONNX format

## Inference Example
### Input
<img src="demo_data/input1.png" alt="Input image">

- **Version**: Single RGB image (224×224) of a lemon.
- **Language**: Describe this image.

### Output
```
This image shows a single, yellow, spherical object that appears to be a small, smooth, and rounded lemon. It is placed on a light-colored, possibly white or off-white, surface with a wood grain texture. The lemon has a rounded, slightly flattened top and a smooth surface. The lighting is even, and the object is the central focus of the image.
```

## Requirements
- Will be supplemented.

## Next task
- Adapt images of different sizes
- Comparison of Test Torch and ONNX inference Speed
- Convert ONNX to TensorRT to further improve inference speed
- Convert more models from Torch to ONNX


## Usage
### 1. Download Qwen3-VL
```bash
# Download the model
hf download Qwen/Qwen3-VL-2B-Instruct
```

### 2. Conert Torch to ONNX and Test Inference
```bash
python qwen3_vl_export_onnx.py
python inference_onnx.py
```

### 3. Conert ONNX to TensorRT and Test Inference
```bash
bash build_engine.sh
python inference_trt.py
```

## Model Conversion Notes
- The ONNX model is exported from the original PyTorch implementation of qwen3-vl-2b.
- Input resolution is fixed at 224×224 (consistent with the model's training configuration).
- For optimal performance, use ONNX Runtime with GPU acceleration (install `onnxruntime-gpu` instead of `onnxruntime`).
- The model retains the original qwen3-vl-2b's visual understanding and text generation capabilities.


## Performance Benchmark
| **Metric** | **Type** | **Value** |
|------------|----------|-----------|
| **Latency (1000 runs)** | Torch (fp32) | 44.46 (sec) |
| | ONNX (fp32) | 26.78 (sec) |
| | ONNX (fp16) | 18.13 (sec) |
| | TensorRT (fp16) | 13.77 (sec) |
| **Generation Speed (10 runs, fp16)** | Qwen3-vl | 19.378385 (tokens/sec) <br>*(Tokens generated: 1103)* |
| | ONNX (tokens/sec) | 38.667467 (tokens/sec) <br>*(Tokens generated: 1062)* |
| | TensorRT (tokens/sec) | 66.579019 (tokens/sec) <br>*(Tokens generated: 842)* |



## License
The model is licensed under the same license as the original qwen3-vl-2b (see [Qwen Official Repository](https://github.com/QwenLM/Qwen) for details).

## Acknowledgements
- Original qwen3-vl-2b model developed by Alibaba Cloud.
- ONNX conversion leverages PyTorch's `torch.onnx.export` API and ONNX Runtime for inference optimization.
