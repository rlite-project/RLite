import subprocess

import rlite
from rlite import RliteInferenceEngine

if __name__ == "__main__":
    rlite.init()

    model_name_or_path = "Qwen/Qwen2.5-7B-Instruct"
    engine = RliteInferenceEngine(model_name_or_path, executor="vllm")
    engine.build(tensor_parallel_size=4)  # On GPU

    engine.release_memory_cache(gpu=True, cpu=False)

    print(f"After building, device: \033[92m{engine.device()}\033[0m")

    subprocess.run(["nvidia-smi"])

    engine.cpu()  # On CPU.

    print(f"After cpu(), device: \033[92m{engine.device()}\033[0m")

    subprocess.run(["nvidia-smi"])

    engine.cuda()  # On GPU again.

    print(f"After cuda(), device: \033[92m{engine.device()}\033[0m")

    subprocess.run(["nvidia-smi"])

    engine.meta()

    print(f"After meta(), device: \033[92m{engine.device()}\033[0m")

    subprocess.run(["nvidia-smi"])
