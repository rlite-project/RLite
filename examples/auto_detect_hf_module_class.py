import transformers
from accelerate import init_empty_weights
from transformers import AutoConfig

import rlite
import rlite.nn
from rlite import RliteTrainEngine

if __name__ == "__main__":
    rlite.init()

    config = AutoConfig.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        attn_implementation="flash_attention_2"
    )
    actor_cls = getattr(transformers, config.architectures[0])

    # Construct a class based on the config
    class Actor(actor_cls, rlite.nn.HuggingFaceFsdp2TrainModule):
        ...

    with init_empty_weights():
        module = Actor(config)
    engine = RliteTrainEngine(module, executor="fsdp2")
    engine.build(tensor_parallel_size=4)
