import rlite
from rlite import RliteInferenceEngine

if __name__ == "__main__":
    rlite.init(suppress_vllm_logging=True)

    model_name_or_path = "Qwen/Qwen2.5-7B-Instruct"
    prompts = [
        "你好，请介绍一下你自己。",
        "Hello, how are you?",
        "中国的首都是",
        "France is famous for its",
    ]

    engine = RliteInferenceEngine(model_name_or_path, executor="vllm")
    engine.build(tensor_parallel_size=4)

    sampling_params = {"temperature": 0.7, "seed": 42}
    outputs = engine.generate(prompts, sampling_params=sampling_params)

    for prompt, output in zip(prompts, outputs):
        print(f"\033[92m{prompt}\033[94m{output.outputs[0].text}\033[0m")
