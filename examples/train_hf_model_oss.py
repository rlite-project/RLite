import torch
import torch.nn.functional as F
import torch.optim
from transformers import GptOssConfig, GptOssForCausalLM, Mxfp4Config

import rlite
import rlite.nn
from rlite import NeedParallel, RliteTrainEngine


class MyGptOssModel(rlite.nn.HuggingFaceFsdp2TrainModule, GptOssForCausalLM):
    def on_weights_materialized(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)

    def train_step(self, input_ids: list[list[int]], gradient_accumulation_steps: int = 1):
        input_ids = torch.tensor(input_ids, dtype=torch.long, device="cuda")

        logits = self(input_ids).logits[:, :-1].contiguous()
        labels = input_ids[:, 1:].contiguous()

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1)).mean()
        loss = loss / gradient_accumulation_steps
        loss.backward()

        return loss.item()

    def optimizer_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()


if __name__ == "__main__":
    rlite.init()

    with rlite.device("meta"):
        quantization_config = Mxfp4Config(dequantize=True)
        config = GptOssConfig.from_pretrained(
            "openai/gpt-oss-20b",
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            use_cache=False,
        )
        module = MyGptOssModel(config)
        module.gradient_checkpointing_enable()

    engine = RliteTrainEngine(module)
    engine.build(tensor_parallel_size=8)

    num_gpus = 8
    gradient_accumulation_steps = 4

    losses = []

    for _ in range(gradient_accumulation_steps):
        input_ids = torch.randint(0, 10000, (num_gpus, 1024))
        losses_for_each_gpu = engine.train_step(
            NeedParallel(input_ids),
            gradient_accumulation_steps=gradient_accumulation_steps
        )
        losses.append(losses_for_each_gpu)

    print(losses)
