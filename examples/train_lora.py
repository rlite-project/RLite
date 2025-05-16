from functools import cached_property

import torch
from transformers import Qwen2ForCausalLM

import rlite
import rlite.nn

MODE = "lora"
# MODE = "full"


class ModelForTest(rlite.nn.HuggingFaceFsdp2TrainModule, Qwen2ForCausalLM):
    def on_weights_materialized(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)

    def get_memory_usage(self):
        torch.cuda.empty_cache()
        return torch.cuda.memory_allocated() // 1024 ** 3  # In GB

    def run_test(self):
        print(f"{MODE} mode: Before running", self.get_memory_usage())
        input_ids = torch.randint(0, 10000, (8, 1024), device="cuda")
        loss = self(input_ids).logits.mean()
        loss.backward()
        self.optimizer.step()
        print(f"{MODE} mode: After running", self.get_memory_usage())

    def preprocess_materialization_source(self, name: str, param: torch.Tensor):
        if not hasattr(self, "weight_mapping"):
            raise ValueError("weight_mapping is not set")
        if name in self.weight_mapping:
            name = self.weight_mapping[name]
        return name, param


class LoraTest:
    def run(self):
        rlite.init()
        self.model.run_test()

    @cached_property
    def model(self):
        from accelerate import init_empty_weights

        with init_empty_weights():
            model = ModelForTest.from_pretrained(
                "Qwen/Qwen2.5-7B",
                attn_implementation="flash_attention_2"
            )
            model.gradient_checkpointing_enable()
            weight_mapping = {}
            if MODE == "lora":
                # We need to handle the changed naming of the keys.
                # The rules are:
                #   1. for all keys, prepend "base_model.model."
                #   2. for the linear layers, add "base_layer." before "weight"
                # We use a weight mapping here to achieve this.
                orig_state_dict = model.state_dict()
                model = self._apply_lora(model)
                new_state_dict = model.state_dict()
                for key in new_state_dict.keys():
                    orig_key = self._get_orig_key(key)
                    if orig_key in orig_state_dict:
                        weight_mapping[orig_key] = key
            model.weight_mapping = weight_mapping

        engine = rlite.RliteTrainEngine(model)
        engine.build(tensor_parallel_size=8)
        return engine

    def _get_orig_key(self, key: str):
        return key.replace("base_layer.", "").replace("base_model.model.", "")

    def _apply_lora(self, model):
        from peft import LoraConfig, TaskType, get_peft_model

        peft_config = LoraConfig(
            r=32,
            lora_alpha=128,
            target_modules="all-linear",
            lora_dropout=0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config)
        return model


if __name__ == "__main__":
    lora_test = LoraTest()
    lora_test.run()
