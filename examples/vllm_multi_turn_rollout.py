from ray.util.queue import Queue

import rlite
from rlite import RliteInferenceEngine
from rlite.inference import IndexedGenerationHistory


def post_generation_hook(
    generation: IndexedGenerationHistory,
    input_queue: Queue,
    output_queue: Queue
):
    """Here, we rollout for twice."""
    if len(generation.outputs) < 2:
        new_input_text = generation.inputs[-1] + generation.outputs[-1].outputs[0].text
        new_generation = IndexedGenerationHistory(
            index=generation.index,
            inputs=generation.inputs + [new_input_text],
            outputs=generation.outputs
        )
        input_queue.put(new_generation)
    else:
        output_queue.put(generation)


if __name__ == "__main__":
    rlite.init()

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
    outputs = engine.generate(
        prompts,
        sampling_params=sampling_params,
        post_generation_hook=post_generation_hook,
    )

    for prompt, output in zip(prompts, outputs):
        print(output)
