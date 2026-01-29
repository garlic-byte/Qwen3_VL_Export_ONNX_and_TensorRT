
from transformers import AutoProcessor


def get_model_input(config):
    processor = AutoProcessor.from_pretrained(config.qwen_path)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "demo_data/input1.png",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    # Check context
    assert len(messages) == config.batch_size, f"messages number should be equal to batch_size, but now messages batch size = {len(messages)}, config batch_size = {config.batch_size}"
    for i in range(config.batch_size):
        numbers_img = sum([1 if content["type"] == "image" else 0 for content in messages[i]["content"]])
        # assert numbers_img == config.imgs_nums, f"The number of imgs is mismatch, {numbers_img} != {config.imgs_nums}"

    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(config.device)