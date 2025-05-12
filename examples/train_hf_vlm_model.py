import torch
from accelerate import init_empty_weights
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration
)

import rlite
import rlite.nn
from rlite import NeedParallel, RliteTrainEngine


def prepare_image(image_url: str):
    from io import BytesIO

    import requests
    from PIL import Image

    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image


class MyQwenVLModel(rlite.nn.HuggingFaceFsdp2TrainModule, Qwen2_5_VLForConditionalGeneration):
    def on_weights_materialized(self):
        # NOTE: This will place the processor on the worker process, which
        #       reduces the communication cost between the main process and
        #       the worker processes.
        model_name_or_path = self.materialization_source()
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, use_fast=True)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)

    def train_step(
        self,
        messages: list[list[dict]],
        image_urls: list[str],
        gradient_accumulation_steps: int = 1
    ):
        prompt_texts = [
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in messages
        ]
        images = [prepare_image(image_url) for image_url in image_urls]
        inputs = self.processor(
            text=prompt_texts,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to("cuda")
        logits = self(**inputs).logits
        # Your loss function here, based on the logits
        loss = logits.mean() / gradient_accumulation_steps
        loss.backward()

    def optimizer_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()


if __name__ == "__main__":
    rlite.init()

    with init_empty_weights():
        config = AutoConfig.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            attn_implementation="flash_attention_2"
        )
        model = MyQwenVLModel(config)
        model.gradient_checkpointing_enable()

    engine = RliteTrainEngine(model, executor="fsdp2")
    engine.build(tensor_parallel_size=4)

    num_gpus = 8
    gradient_accumulation_steps = 4

    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "https://picsum.photos/id/1/200/300"},
                    {"type": "text", "text": "Describe the image in detail."}
                ]
            }
        ]
    ] * num_gpus
    image_urls = ["https://picsum.photos/id/1/200/300" for _ in range(num_gpus)]

    for _ in range(gradient_accumulation_steps):
        engine.train_step(
            NeedParallel(messages),
            NeedParallel(image_urls),
            gradient_accumulation_steps=gradient_accumulation_steps
        )
    engine.optimizer_step()
