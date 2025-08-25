import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.distributed.tensor import DTensor
from transformers import Qwen2Config, Qwen2ForCausalLM

import rlite
import rlite.nn
from rlite import RliteInferenceEngine, RliteTrainEngine


class MyModel(rlite.nn.HuggingFaceFsdp2TrainModule, Qwen2ForCausalLM):
    def on_weights_materialized(self):
        self._init_lora_B_as_zeros()

    def preprocess_input_named_parameters(
        self,
        name: str,
        param: torch.Tensor,
    ) -> tuple[str, torch.Tensor]:
        if not hasattr(self, "weight_mapping"):
            raise ValueError("weight_mapping is not set")
        if name in self.weight_mapping:
            name = self.weight_mapping[name]
        return name, param

    def postprocess_output_named_parameters(
        self,
        name: str,
        param: DTensor,
        full_state_dict: dict[str, DTensor]
    ) -> tuple[str, torch.Tensor]:
        if "base_layer.weight" in name:
            lora_a = full_state_dict[name.replace("base_layer.", "lora_A.default.")]
            lora_b = full_state_dict[name.replace("base_layer.", "lora_B.default.")]
            param = param.full_tensor() + lora_b.full_tensor() @ lora_a.full_tensor()
        else:
            param = param.full_tensor()
        name = name.replace("base_model.model.", "").replace("base_layer.", "")
        return name, param

    def _init_lora_B_as_zeros(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                if "lora_B" in name:
                    param.zero_()


def apply_lora(model: Qwen2ForCausalLM) -> torch.nn.Module:
    peft_confg = LoraConfig(
        r=32,
        lora_alpha=128,
        target_modules="all-linear",
        lora_dropout=0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    weight_mapping = {}
    orig_state_dict = model.state_dict()
    model = get_peft_model(model, peft_confg)
    new_state_dict = model.state_dict()
    for key in new_state_dict.keys():
        orig_key = key.replace("base_model.model.", "").replace("base_layer.", "")
        if orig_key in orig_state_dict:
            weight_mapping[orig_key] = key
    model.weight_mapping = weight_mapping

    return model


if __name__ == "__main__":
    rlite.init()

    model_name_or_path = "Qwen/Qwen2.5-7B-Instruct"

    vllm_engine = RliteInferenceEngine(model_name_or_path, executor="vllm")
    vllm_engine.build(tensor_parallel_size=4)

    vllm_engine.meta()  # Drop all the weights

    with rlite.device("meta"):
        config = Qwen2Config.from_pretrained(model_name_or_path)
        model = apply_lora(MyModel(config))

    fsdp2_engine = RliteTrainEngine(model, executor="fsdp2")
    fsdp2_engine.build(tensor_parallel_size=8, colocate_with=vllm_engine)

    print("Syncing weights from VLLM to FSDP2", flush=True)
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
