import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from transformers import Qwen2ForCausalLM

import rlite
import rlite.nn
from rlite import NeedParallel, RliteInferenceEngine, RliteTrainEngine


class MyQwenModel(rlite.nn.HuggingFaceFsdp2TrainModule, Qwen2ForCausalLM):
    def on_weights_materialized(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)

    def train_step(self, input_ids: list[list[int]], grad_acc_steps: int = 1):
        input_ids = torch.tensor(input_ids, dtype=torch.long, device="cuda")

        # Language loss
        logits = self(input_ids).logits[:, :-1].contiguous()
        labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1)).mean()

        # Backward and gradient accumulation
        loss = loss / grad_acc_steps
        loss.backward()

        return loss.item()

    def optimizer_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()


if __name__ == "__main__":
    rlite.init()
    vllm_engine = RliteInferenceEngine("Qwen/Qwen2.5-7B-Instruct", executor="vllm")
    vllm_engine.build(tensor_parallel_size=4)

    prompts = ["你好，世界！", "Hello, world!"] * 8
    rollouts = vllm_engine.generate(prompts)

    vllm_engine.meta()  # Release everything from GPU

    with init_empty_weights():
        module = MyQwenModel.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    fsdp2_engine = RliteTrainEngine(module, executor="fsdp2")
    fsdp2_engine.build(tensor_parallel_size=4, colocate_with=vllm_engine)

    input_ids = [[x[0].outputs[0].token_ids for x in rollouts] for _ in range(4)]
    grad_acc_steps = 4

    for step in range(grad_acc_steps):
        fsdp2_engine.train_step(
            NeedParallel(input_ids[step]),  # This will be split among the workers (DP)
            grad_acc_steps  # This will be copied to all workers
        )
    fsdp2_engine.optimizer_step()

    vllm_engine.cuda("weights")  # Only the weights are loaded
    fsdp2_engine.p2p_weight_sync(vllm_engine)  # GPU-to-GPU weight sync via CUDAIPC
    fsdp2_engine.cpu()
    vllm_engine.cuda("kv_cache")  # The KV cache of vLLM is back to GPU
