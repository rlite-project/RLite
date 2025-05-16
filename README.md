# RLite

**RLite** is a lightweight reinforcement learning framework that integrates seamlessly into your codebase, empowering developers to focus on algorithms with minimal intrusion.

## Installation

We recommend using `conda` to manage our computation environment.

1. Create a conda environment:

```
conda create -n rlite python==3.12
conda activate rlite
```

2. Install common dependencies

```bash
# install vllm
pip install vllm accelerate

# flash attention 2 (make sure you have 64 CPU cores)
MAX_JOBS=64 pip install flash-attn --no-build-isolation

# Install fashinfer for faster inference
pip install flashinfer-python==0.2.2.post1 -i https://flashinfer.ai/whl/cu124/torch2.6
```

4. Install `rlite`

```bash
git clone https://github.com/rlite-project/RLite.git
cd RLite; pip install -e .
```

## Quick Start: Write an RL sketch in 10 minutes

In this tutorial, we will use a simplified example to show you how to write an RL program with `rlite` in tens of lines, while still scales well.

### 1. Import packages

First, import `rlite` as a common python package.

```python
import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from transformers import Qwen2ForCausalLM

import rlite
import rlite.nn
from rlite import NeedParallel, RliteInferenceEngine, RliteTrainEngine
```

### 2. Define your `torch.nn.Module`

Then, define a pytorch module that implements the logic of your algorithm. Here we use the language loss for simplicity.

```python
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
```

Here we are subclassing the `Qwen2ForCausalLM` from `transformers` to reuse its forward implementation. We also subclass `rlite.nn.HuggingFaceFsdp2TrainModule`, which is basically a `torch.nn.Module` that further implements

1. serialization of the module, so that this module can be sent/received using `ray`;
2. necessary interfaces (e.g., get the checkpoint path) for materializing and training the module on the cluster.

> **_NOTE:_** In `rlite`, we use `rlite.nn.xxxTrainModule` as the carrier for user-defined logics. Such modules **must** be initialized on torch's meta device to ensure efficient module transfer between processes (i.e. ray actors). This enables algorithm-computation decoupling, allowing users to focus on algorithms only (writing `nn.Module`s). `rlite` will handle everything else.

> **_NOTE:_** `on_weights_materialized` is an important hook for operations that are expected to happen after the module is materialized on GPUs. For example, this is useful for optimizer and LR schedulers. You don't need to implement this hook if your module does not need to be optimized, e.g. reference model in PPO.

### 3. Initialize `rlite`

Call `rlite.init()` to config `rlite` globally and initialize the resource manager.

```python
rlite.init()
```

### 4. Initialize a vLLM actor for inference

Then, we initialize a vLLM actor and use it to generate rollouts.

```python
vllm_engine = RliteInferenceEngine("Qwen/Qwen2.5-7B-Instruct", executor="vllm")
vllm_engine.build(tensor_parallel_size=4)

prompts = ["你好，世界！", "Hello, world!"] * 8
rollouts = vllm_engine.generate(prompts)
```

### 5. Initialize a FSDP2 actor for training

After generation, we drop all the weights of this vLLM actor and initialize the train actor. Note that the module **must** be initialized on `meta` device to get a low-memory module instance.

```python
vllm_engine.meta()  # Release everything from GPU

with init_empty_weights():
    module = MyQwenModel.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
fsdp2_engine = RliteTrainEngine(module, executor="fsdp2")
fsdp2_engine.build(tensor_parallel_size=4, colocate_with=vllm_engine)
```

> **_NOTE:_** You can use `.cpu()`, `.cuda(*args, **kwargs)`, or `.meta()`, to move the engine to CPU, to GPU, or release all memory usage in both CPU and GPU.

### 6. Call the user-defined training logic

Let's start training!

```python
input_ids = [[x.outputs[0].token_ids for x in rollouts] for _ in range(4)]
grad_acc_steps = 4

for step in range(grad_acc_steps):
    fsdp2_engine.train_step(
        NeedParallel(input_ids[step]),  # This will be split among the workers (DP)
        grad_acc_steps  # This will be copied to all workers
    )
fsdp2_engine.optimizer_step()
```

As you can see, you can call the function you just defined through the `RliteTrainEngine`. This gives the full flexibility to users to design new algorithms, without worrying about adapting them to `rlite`.

### 7. Sync weight from FSDP2 actor to vLLM actor

After training, sync the updated weights to vLLM actor and generate again.

```python
vllm_engine.cuda("weights")  # Only the weights are loaded
fsdp2_engine.p2p_weight_sync(vllm_engine)  # GPU-to-GPU weight sync via CUDAIPC
fsdp2_engine.cpu()
vllm_engine.cuda("kv_cache")  # The KV cache of vLLM is back to GPU
```

That's all 🎉! Writing an RL program should be simple 😄! Check out the [examples](https://github.com/rlite-project/RLite/tree/main/examples) for atomic usage examples and [recipes](https://github.com/rlite-project/RLite-Recipe) for complete examples that reproduce state-of-the-art RL results.

## Contributing

<details>
<summary>Developer's guide.</summary>

<br>

> Write code that you would like to read again.

We use `pre-commit` and `git cz` to sanitize the commits. You can run `pre-commit` before `git cz` to avoid repeatedly input the commit messages.

```bash
pip install pre-commit
# Install pre-commit hooks
pre-commit install
pre-commit install --hook-type commit-msg
# Install this emoji-style tool
sudo npm install -g git-cz --no-audit --verbose --registry=https://registry.npmmirror.com

# Install rlite
pip install -e ".[dev]"
```

##### Code Style

- Single line code length is 99 characters, comments and documents are 79 characters.
- Write unit tests for atomic capabilities to ensure that `pytest` does not throw an error.

Run `pre-commit` to automatically lint the code:

```
pre-commit run --all-files
```

##### Run Unit Tests:

```bash
# Only run tests
pytest

# Run tests and output test code coverage report
pytest --cov=rlite
```

</details>
