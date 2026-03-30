
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