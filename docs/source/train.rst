Training with RLite
===================

In RLite, model training is very similar to PyTorch. Users need to define a ``rlite.nn.BaseTrainModule`` subclass and implement the computational logic as methods. Then, users can call the methods they defined anywhere they want.

RLite's training engines (FSDP2, FSDP1) make no assumptions about model architecture. However, as deep learning model implementations have gradually converged, users often prefer to build their own models starting from proven architectures. Therefore, we will introduce how to train common model architectures in RLite, as well as how to train your own custom models.

.. note::

    In RLite, FSDP2 backend does not support calling ``self.forward(*args, **kwargs)`` in custom methods, as FSDP2 will override the ``forward`` method. Use ``self(*args, **kwargs)`` instead.

Train with HuggingFace Language Models
--------------------------------------

Training HuggingFace language models in RLite is straightforward, with a programming experience similar to traditional PyTorch. Users first define their own Module class, then feed the correct data into the training process and execute backpropagation and gradient descent.

Unlike bare PyTorch, in RLite, training modules need to inherit from ``rlite.nn.BaseHuggingFaceTrainModule`` instead of ``torch.nn.Module``. This interface inherits from ``torch.nn.Module`` and adds interfaces for data transfer and training. In RLite, these modules carry the user's algorithmic logic, and RLite automatically runs this logic in parallel on specific computing resources.

Before being instantiated on actual computing resources, user-defined modules are placed on the meta device by default, allowing them to be efficiently serialized and transferred to other machines. For HuggingFace models, users can use the ``init_empty_weights`` context method from ``accelerate`` to initialize the model.

.. note::

    Using ``torch.device("meta")`` can be problematic because models created this way will have zero-initialized buffers, which can cause layers with buffers like RoPE to malfunction.

After calling the engine's ``build`` method, modules are moved to specific computing resources and initialized. Once initialization is complete, users can bind post-processing logic in the ``on_weights_materialized`` method. For HuggingFace models that don't require training (such as reference models), RLite generally doesn't require users to implement any interfaces. For models that need parameter updates, users need to bind optimizers in the ``on_weights_materialized`` method.

Users can also define custom methods, which can be called after the engine has been ``build``.

.. code-block:: python

   from transformers import Qwen2Config, Qwen2ForCausalLM

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

    rlite.init()
    with init_empty_weights():
        module = MyQwenModel.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    train_engine = RliteTrainEngine(module, executor="fsdp2")
    train_engine.build(tensor_parallel_size=4)

    losses_for_each_gpu = train_engine.train_step(
        NeedParallel([[1, 2, 3, 4] * 8]),
        gradient_accumulation_steps=4
    )
    train_engine.optimizer_step()

Here, ``NeedParallel`` is a helper class that doesn't do anything by itself; it simply marks that the data needs to be processed in parallel across multiple executors (where the first dimension is the batch_size dimension, which can be independently processed by multiple model instances). This class can also be used to mark data that needs to be sequence-parallelized:

.. code-block:: python

    seq_parallel_data = NeedParallel([[1, 2, 3, 4] * 8], type="seq")

.. note::

   In RLite's programming model, the parameter and return value transfer is handled by Ray, which may become slow when handling large amounts of data. Users should be aware of this and avoid unnecessary large data transfers whenever possible. For example, users can place data processing logic in worker processes by defining methods in ``rlite.nn.BaseTrainModule`` to perform data processing, and send the data URL between the main process and workers.


Train with HuggingFace Vision-Language Models
---------------------------------------------

Similar to language models, when training vision-language models, users need to first define a vision-language nn.Module. Taking ``Qwen2_5_VLForConditionalGeneration`` as an example:

.. code-block:: python

    class MyQwenVLModel(rlite.nn.HuggingFaceFsdp2TrainModule, Qwen2_5_VLForConditionalGeneration):
        def on_weights_materialized(self):
            # NOTE: This will place the processor on the worker process, which
            #       reduces the communication cost between the main process and
            #       the worker processes.
            model_name_or_path = self.materialization_source()
            self.processor = AutoProcessor.from_pretrained(model_name_or_path, use_fast=True)

            self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)

        def train_step(
            self,
            messages: list[list[dict]],
            image_urls: list[str],
            gradient_accumulation_steps: int = 1
        ):
            prompt_texts = [
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for messages in messages
            ]
            images = [prepare_image(image_url) for image_url in image_urls]
            inputs = self.processor(
                text=prompt_texts,
                images=images,
                return_tensors="pt",
                padding=True,
            ).to("cuda")
            logits = self(**inputs).logits
            # Your loss function here, based on the logits
            loss = logits.mean() / gradient_accumulation_steps
            loss.backward()

        def optimizer_step(self):
            self.optimizer.step()
            self.optimizer.zero_grad()

Unlike language models, we can implement the data processing logic within the module's methods, which reduces the data transfer cost and avoids passing image data over Ray. For specific forward pass logic and data inputs, please refer to HuggingFace's transformers library.

Train with HuggingFace ``peft`` Models
--------------------------------------

For scenarios with limited computational resources, users may want to use the ``peft`` library for lightweight fine-tuning. The ``peft`` library wraps the user's module, changing the model's state_dict. Additionally, the inference engine needs to accept parameters after applying the peft model's adapter (such as applying LoRA) for inference.

In RLite, we provide two interfaces to support training ``peft`` models. These primarily support hooks for processing the state_dict during model initialization and export.

.. code-block:: python

    class LoraTrainModule(rlite.nn.HuggingFaceFsdp2TrainModule):
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


For more details, please refer to `this example <https://github.com/rlite-project/RLite-Recipe/tree/main/recipe/grpo_lora>`_ which demonstrates the implementation.


Train with Custom Models
------------------------

RLite supports training arbitrary custom models. Unlike HuggingFace models, we need to implement some key interfaces to support initialization and parallelization.

.. code-block:: python

    class MyModel(rlite.nn.Fsdp2TrainModule):
        @property
        def non_split_modules(self) -> set[type[torch.nn.Module]]:
            ...

        def materialization_source(self) -> dict[str, torch.Tensor] | dict[str, callable] | str:
            ...

The ``materialization_source`` method provides either a state_dict for initializing model parameters, a function for initializing the state_dict, or a path to a checkpoint file from which a complete state_dict can be loaded.

The ``non_split_modules`` property returns a set of module types that should not be split across FSDP instances.
