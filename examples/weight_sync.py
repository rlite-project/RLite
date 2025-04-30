from accelerate import init_empty_weights
from transformers import Qwen2ForCausalLM

import rlite
import rlite.nn
from rlite import RliteInferenceEngine, RliteTrainEngine


class MyModel(rlite.nn.HuggingFaceFsdp2TrainModule, Qwen2ForCausalLM):
    ...


if __name__ == "__main__":
    rlite.init()

    # model_name_or_path = "Qwen/Qwen2.5-7B-Instruct"
    model_name_or_path = "/mnt/o1-reproduce/checkpoints/Qwen2.5-7B-Instruct"

    vllm_engine = RliteInferenceEngine(model_name_or_path, executor="vllm")
    vllm_engine.build(tensor_parallel_size=4)

    vllm_engine.meta()  # Drop all the weights

    with init_empty_weights():
        model = MyModel.from_pretrained(model_name_or_path)

    fsdp2_engine = RliteTrainEngine(model, executor="fsdp2")
    fsdp2_engine.build(tensor_parallel_size=8, colocate_with=vllm_engine)

    vllm_engine.cuda("weights")
    fsdp2_engine.p2p_weight_sync(vllm_engine)
    fsdp2_engine.cpu()  # Offload the weights to CPU
    vllm_engine.cuda("kv_cache")

    prompts = [
        "你好，请介绍一下你自己。",
        "Hello, how are you?",
        "中国的首都是",
        "France is famous for its",
    ]
    print(vllm_engine.generate(prompts))
