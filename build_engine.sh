# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

export PATH=/usr/src/tensorrt/bin:$PATH

# Define length variables
MIN_LEN=50
OPT_LEN=500
MAX_LEN=1024
BATCH_SIZE=1
IMG_NUMS=1
IMG_LENGTH=512
ONNX_DIR="export/qwen3_vl_2b"


if [ -e /usr/src/tensorrt/bin/trtexec ]; then
    echo "The file /usr/src/tensorrt/bin/trtexec exists."
else
    echo "The file /usr/src/tensorrt/bin/trtexec does not exist. Please install tensorrt"
fi

mkdir -p ${ONNX_DIR}/tensorrt

echo "------------Building LLM Model--------------------"
trtexec --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers \
--onnx=${ONNX_DIR}/ONNX/llm/llm.onnx --saveEngine=${ONNX_DIR}/tensorrt/llm.engine \
--minShapes=position_ids:3x${BATCH_SIZE}x${MIN_LEN},inputs_embeds:${BATCH_SIZE}x${MIN_LEN}x2048,visual_pos_masks:${BATCH_SIZE}x${MIN_LEN},deepstack_visual_embeds:3x${MIN_LEN}x2048 \
--optShapes=position_ids:3x${BATCH_SIZE}x${OPT_LEN},inputs_embeds:${BATCH_SIZE}x${OPT_LEN}x2048,visual_pos_masks:${BATCH_SIZE}x${OPT_LEN},deepstack_visual_embeds:3x${OPT_LEN}x2048 \
--maxShapes=position_ids:3x${BATCH_SIZE}x${MAX_LEN},inputs_embeds:${BATCH_SIZE}x${MAX_LEN}x2048,visual_pos_masks:${BATCH_SIZE}x${MAX_LEN},deepstack_visual_embeds:3x${MAX_LEN}x2048 \
> ${ONNX_DIR}/tensorrt/llm.log 2>&1


echo "------------Building VIT Model--------------------"
trtexec --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers \
--onnx=${ONNX_DIR}/ONNX/vit/vit.onnx --saveEngine=${ONNX_DIR}/tensorrt/vit.engine \
--minShapes=hidden_states:256x1536,image_grid_thw:1x3 \
--optShapes=hidden_states:256x1536,image_grid_thw:2x3 \
--maxShapes=hidden_states:256x1536,image_grid_thw:2x3 \
> ${ONNX_DIR}/tensorrt/vit.log 2>&1


echo "------------Building VLM Model--------------------"
trtexec --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers \
--onnx=${ONNX_DIR}/ONNX/vlm/vlm.onnx --saveEngine=${ONNX_DIR}/tensorrt/vlm.engine \
--minShapes=input_ids:1x${MIN_LEN},attention_masks:1x${MIN_LEN},image_embeds:64x2048 \
--optShapes=input_ids:1x${OPT_LEN},attention_masks:1x${OPT_LEN},image_embeds:64x2048 \
--maxShapes=input_ids:1x${MAX_LEN},attention_masks:1x${MAX_LEN},image_embeds:64x2048 \
> ${ONNX_DIR}/tensorrt/vlm.log 2>&1


echo "------------Building GEN Model--------------------"
trtexec --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers \
--onnx=${ONNX_DIR}/ONNX/gen/gen.onnx --saveEngine=${ONNX_DIR}/tensorrt/gen.engine \
--minShapes=hidden_states:1x${MIN_LEN}x2048  \
--optShapes=hidden_states:1x${OPT_LEN}x2048 \
--maxShapes=hidden_states:1x${MAX_LEN}x2048 \
> ${ONNX_DIR}/tensorrt/gen.log 2>&1