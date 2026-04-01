import torch
from transformers import AutoProcessor


def get_model_input(config):
    processor = AutoProcessor.from_pretrained(config.qwen_path)
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": img_path} for img_path in config.imgs_paths]
                + [{"type": "text", "text": "Describe this image."}],
        }
    ]

    # Check context
    assert len(messages) == config.batch_size, f"messages number should be equal to batch_size, but now messages batch size = {len(messages)}, config batch_size = {config.batch_size}"

    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(config.device)


def get_qwen3_onnx_input(config, torch_input, llm_hidden_size, vit_hidden_size, gen_hidden_size):
    """
    torch_input: dict contains: input_ids, pixel_values, image_grid_thw
    return: inputs of different part model
    """
    input_ids = torch_input["input_ids"].clone()    # shape torch.Size([1, 144])
    batch_size, seq_len = input_ids.shape
    deepstack_visual_len = 3

    position_ids = torch.ones(
        (3, batch_size, seq_len), dtype=torch.int64
    ).to(config.device) # torch.Size([3, 1, 144])

    inputs_embeds = torch.zeros(
        (batch_size, seq_len, llm_hidden_size), dtype=config.dtype
    ).to(config.device) # torch.Size([1, 144, 2048])

    visual_pos_masks = torch.rand(batch_size, seq_len) > 0.5
    x = visual_pos_masks.sum().item()
    visual_pos_masks = visual_pos_masks.to(config.device) # torch.Size([1, 144])

    deepstack_visual_embeds = torch.randn(
        (deepstack_visual_len, x, llm_hidden_size), dtype=config.dtype
    ).to(config.device) # torch.Size([3, 67, 2048])


    hidden_states = torch_input["pixel_values"].clone().to(dtype=config.dtype) # seq_len x 1536
    image_grid_thw = torch_input["image_grid_thw"].clone() # img_num x 3

    attention_masks = torch_input["attention_mask"].clone()    # shape torch.Size([1, 144])
    image_embeds = torch.randn(
        (64, vit_hidden_size), dtype=config.dtype
    )

    gen_hidden_states = torch.randn(
        (batch_size, seq_len, gen_hidden_size), dtype=config.dtype
    ).to(config.device)


    qwen3_onnx_inputs = {
        "llm": {
            "inputs": (position_ids, inputs_embeds, visual_pos_masks, deepstack_visual_embeds),
            "input_names": ["position_ids", "inputs_embeds", "visual_pos_masks", "deepstack_visual_embeds"],
            "output_names": ["hidden_states"],
            "dynamic_axes": {
                "position_ids": {1: "batch_size", 2: "seq_length"},
                "inputs_embeds": {0: "batch_size", 1: "seq_length"},
                "visual_pos_masks": {0: "batch_size", 1: "seq_length"},
                "deepstack_visual_embeds": {1: "visual_seqlen"},
                "hidden_states": {0: "batch_size", 1: "seq_length"},
            }
        },
        "vit": {
            "inputs": (hidden_states, image_grid_thw),
            "input_names": ["hidden_states", "image_grid_thw"],
            "output_names": ["image_embeds", "deepstack_image_embeds"],
            "dynamic_axes": {
                "hidden_states": {0: "seq_len"},
                "image_grid_thw": {0: "img_num"},
            }
        },
        "vlm": {
            "inputs": (input_ids, attention_masks, image_embeds),
            "input_names": ["input_ids", "attention_masks", "image_embeds"],
            "output_names": ["position_ids", "attention_mask", "inputs_embeds", "visual_pos_masks"],
            "dynamic_axes": {
                "input_ids": {0: "batch_size", 1: "seq_length"},
                "attention_masks": {0: "batch_size", 1: "seq_length"},
                "image_grid_thw": {0: "num_images"},
            },
        },
        "gen": {
            "inputs": (gen_hidden_states, ),
            "input_names": ["hidden_states"],
            "output_names": ["logits"],
            "dynamic_axes": {
                "hidden_states": {0: "batch_size", 1: "seq_length"},
            },
        }
    }
    return qwen3_onnx_inputs


def get_qwen35_onnx_input(config, torch_input, llm_hidden_size, vit_hidden_size, gen_hidden_size):
    """
    torch_input: dict contains: input_ids, pixel_values, image_grid_thw
    return: inputs of different part model
    """
    input_ids = torch_input["input_ids"].clone()    # shape torch.Size([1, 144])
    batch_size, seq_len = input_ids.shape
    deepstack_visual_len = 3

    position_ids = torch.ones(
        (4, batch_size, seq_len), dtype=torch.int64
    ).to(config.device) # torch.Size([4, 1, 144])

    inputs_embeds = torch.zeros(
        (batch_size, seq_len, llm_hidden_size), dtype=config.dtype
    ).to(config.device) # torch.Size([1, 144, 2048])

    visual_pos_masks = torch.rand(batch_size, seq_len) > 0.5
    x = visual_pos_masks.sum().item()
    visual_pos_masks = visual_pos_masks.to(config.device) # torch.Size([1, 144])

    deepstack_visual_embeds = torch.randn(
        (deepstack_visual_len, x, llm_hidden_size), dtype=config.dtype
    ).to(config.device) # torch.Size([3, 67, 2048])


    hidden_states = torch_input["pixel_values"].clone().to(dtype=config.dtype) # seq_len x 1536
    image_grid_thw = torch_input["image_grid_thw"].clone() # img_num x 3

    attention_masks = torch_input["attention_mask"].clone()    # shape torch.Size([1, 144])
    image_embeds = torch.randn(
        (64, vit_hidden_size), dtype=config.dtype
    )
    mm_token_type_ids = torch_input["mm_token_type_ids"].clone()     # shape torch.Size([1, 144])

    gen_hidden_states = torch.randn(
        (batch_size, seq_len, gen_hidden_size), dtype=config.dtype
    ).to(config.device)


    qwen35_onnx_inputs = {
        "llm": {
            "inputs": (position_ids, inputs_embeds),
            "input_names": ["position_ids", "inputs_embeds"],
            "output_names": ["hidden_states"],
            "dynamic_axes": {
                "position_ids": {1: "batch_size", 2: "seq_length"},
                "inputs_embeds": {0: "batch_size", 1: "seq_length"},
                "hidden_states": {0: "batch_size", 1: "seq_length"},
            }
        },
        "vit": {
            "inputs": (hidden_states, image_grid_thw),
            "input_names": ["hidden_states", "image_grid_thw"],
            "output_names": ["image_embeds"],
            "dynamic_axes": {
                "hidden_states": {0: "seq_len"},
                "image_grid_thw": {0: "img_num"},
            }
        },
        "vlm": {
            "inputs": (input_ids, attention_masks, image_embeds, mm_token_type_ids, image_grid_thw),
            "input_names": ["input_ids", "attention_masks", "image_embeds", "mm_token_type_ids", "image_grid_thw"],
            "output_names": ["position_ids", "inputs_embeds"],
            "dynamic_axes": {
                "input_ids": {0: "batch_size", 1: "seq_length"},
                "attention_masks": {0: "batch_size", 1: "seq_length"},
                "image_embeds": {1: "seq_length"},
                "mm_token_type_ids": {1: "seq_length"},
                "image_grid_thw": {0: "num_images"},
            },
        },
        "gen": {
            "inputs": (gen_hidden_states, ),
            "input_names": ["hidden_states"],
            "output_names": ["logits"],
            "dynamic_axes": {
                "hidden_states": {0: "batch_size", 1: "seq_length"},
            },
        }
    }
    return qwen35_onnx_inputs