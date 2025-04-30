import torch
import torch.nn.functional as F
import torch.optim
from accelerate import init_empty_weights
from transformers import Qwen2ForCausalLM

import rlite
import rlite.nn
from rlite import NeedParallel, RliteTrainEngine


class MyQwenModel(rlite.nn.HuggingFaceFsdp2TrainModule, Qwen2ForCausalLM):
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

    with init_empty_weights():  # Init on meta device
        module = MyQwenModel.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            attn_implementation="flash_attention_2"
        )
        module.gradient_checkpointing_enable()

    engine = RliteTrainEngine(module, executor="fsdp2")  # Must on meta device
    # NOTE: here tensor_parallel_size means the number of parameter shards
    engine.build(tensor_parallel_size=4)

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
