from transformers import AutoProcessor

import rlite
from rlite import RliteInferenceEngine


def prepare_image(image_url):
    from io import BytesIO

    import requests
    from PIL import Image

    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image


def prepare_prompts(processor, num_prompts: int = 8):
    image_urls = [f"https://picsum.photos/id/{i}/200/300" for i in range(num_prompts)]
    messages = [
        processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_url},
                        {"type": "text", "text": "Describe the image in detail."}
                    ]
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for image_url in image_urls
    ]
    prompts = [
        {
            "prompt": messages[i],
            "multi_modal_data": {
                "image": prepare_image(image_urls[i])
            }
        }
        for i in range(num_prompts)
    ]
    return prompts


if __name__ == "__main__":
    rlite.init()

    model_name_or_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    processor = AutoProcessor.from_pretrained(model_name_or_path, use_fast=True)
    prompts = prepare_prompts(processor, num_prompts=8)

    engine = RliteInferenceEngine(model_name_or_path, executor="vllm")
    engine.build(
        tensor_parallel_size=4,
        max_model_len=4096,
        max_num_seqs=1,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        },
        limit_mm_per_prompt={"image": 1, "video": 0, "audio": 0},
    )

    outputs = engine.generate(prompts, sampling_params={"temperature": 0.7, "seed": 42})
    for output in outputs:
        print(output)
