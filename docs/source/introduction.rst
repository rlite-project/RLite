Introduction
============

Why RLite?
----------

Why develop RLite when the open-source community already has excellent projects like `verl <https://github.com/volcengine/verl>`_ and `openrlhf <https://github.com/OpenRLHF/OpenRLHF>`_?

Existing frameworks have gradually become more complex as they pursue optimal performance and support as many open-source works as possible. This is mainly reflected in the increasing coupling between the framework core code, the underlying parallel engines, and open-source RL recipes. To balance functionality and performance, existing frameworks have had to compromise on flexibility, indirectly increasing the learning cost for new users.

The original intention of RLite is to provide a lightweight framework that makes no compromise on usability and flexibility while not sacrificing performance too much. Existing efficient frameworks are like fully-featured scaffolding, while RLite is more like a Swiss Army knife that allows researchers to quickly experiment with new ideas at reasonable performance levels.

RLite has two main features:

1. PyTorch-like interface design.
2. Algorithm-computation decoupling.

First, the RLite interfaces are designed to be as consistent as possible with PyTorch, as the PyTorch's interface design is already excellent, and it only needs some optimizations behind the scenes to support the RL for large-scale models. In this way, users can enjoy the convenience brought by RLite at nearly no learning cost.

Second, RLite's programming experience is very similar to that of single-process PyTorch programs, and all parallelized computations are hidden inside the framework. This decouples the user's algorithm logic from the specific computational implementation, allowing users to focus on algorithms without worrying too much about performance.

Finally, we also separate RLite from implementing various existing RL algorithms and well-known works, positioning it as a foundational Python library that maintains its lightweight nature and flexibility. RLite will always contain only the minimal core functionality and provide flexible interfaces to support various future works. As a complement, we place useful examples and reproductions of existingcutting-edge works in the RLite-Recipe repository, making them easy for users to reference and use.

Programming Model
-----------------

.. image:: assets/docs/source/intrudoction/program_model.jpg
   :width: 80%
   :align: center

In RLite, users mainly work with **Engine**s, which is a handler that takes the input from the main process, organizes the tasks and sends to the workers. The engine may have multiple **Executor**s, each holding a full set of model weights. Both Engines and Executors reside in the main process. The **Worker**s are the units that actually perform computational tasks, with each Worker corresponding to a GPU. Conversely, a single GPU can be associated with multiple Workers, which can use the GPU in a time-multiplexed manner.

The executor may have different backends, such as ``vLLM`` for inference and ``FSDP2`` for training. The executors of the same engine works in a data-parallel manner. Note that the workers of the same executor may still work in a data-parallel manner, e.g. in FSDP.

This programming model is inspired by ``vLLM`` and is flexible enough to support more advanced parallel backends such as ``megatron-core``. For example, the executor may send the input to the pipeline parallel rank 0 and gather the output from the PP rank -1.

Key Interfaces
--------------

Compared to traditional PyTorch, the fundamental challenge in reinforcement learning for large-scale models today is how to accommodate highly optimized training and inference engines. On-policy learning algorithms require efficient data exchange between training and inference engines, as well as time-multiplexed use of computational resources. Off-policy learning algorithms demand management of heterogeneous computing resources and efficient message passing.

To address these challenges, RLite extends PyTorch's interfaces, enabling users to conveniently manage computational resources and data exchange.

Global Configuration and Resource Management
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``rlite.init()`` function provides global configuration and resource management capabilities. It will attach/initialize the Ray cluster for distributed training and inference.

Users can use this interface to configure the resources used for the entire training process and the underlying engine settings. Refer to :doc:`Interfaces <interfaces>` for more details.

Inference Interface
^^^^^^^^^^^^^^^^^^^

RLite provides a unified interface for all inference engines. Currently, RLite supports the following inference engines:

- `vLLM <https://github.com/vllm-project/vllm>`_

The interface is very simple, just initialize the engine and build with the desired parallelization configuration.

.. code-block:: python

    engine = rlite.RLiteInferenceEngine("Qwen/Qwen2.5-7B-Instruct", executor="vllm")
    engine.build(
        tensor_parallel_size=2,
        # Other args for vLLM
        max_model_len=1024,
        gpu_memory_utilization=0.9,
    )

    outputs: list[vllm.RequestOutput] = engine.generate(
        ["Hello, world!"] * 3,
        sampling_params={
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "max_new_tokens": 10,
        },
    )

By default, the engine will reside on GPU after ``build`` method is called.

Training Interface
^^^^^^^^^^^^^^^^^^

Training in RLite is very similar to PyTorch, where we first define a ``torch.nn.Module`` subclass, whose methods are actual learning operations on GPU. The difference is that in RLite, instead of subclassing ``torch.nn.Module``, users should subclass ``rlite.nn.BaseTrainModule`` or its subclasses such as ``rlite.nn.HuggingFaceFsdp2TrainModule``.

.. code-block:: python

    class MyQwenModel(rlite.nn.HuggingFaceFsdp2TrainModule, Qwen2ForCausalLM):
        def on_weights_materialized(self):
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)

        def train_step(self, input_ids: list[list[int]], gradient_accumulation_steps: int = 1):
            ...

        def optimizer_step(self):
            self.optimizer.step()
            self.optimizer.zero_grad()

    with accelerate.init_empty_weights():
        model = MyQwenModel.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    engine = rlite.RliteTrainEngine(model, executor="fsdp2")
    engine.build(tensor_parallel_size=8)  # FSDP num_shards. num_shards=8 means HSDP

    # Magic! You can call the functions defined in the module on the engine object directly.
    engine.train_step(
        rlite.NeedParallel([[1, 2, 3, 4, 5]] * 8),
        gradient_accumulation_steps=2
    )

The training engine of RLite just takes the user-defined ``rlite.nn.BaseTrainModule`` subclass and sends it to the accelerator. Therefore, this module MUST be initialized on the meta device to support efficient transfer.

As you can see, the code is very similar to single-process PyTorch programs. The train engine will automatically handle data parallel or sequence parallel if the argument is wrapped by ``rlite.NeedParallel``. See :doc:`Train with RLite <train>` for more details.

Device Management Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For both training and inference engines, RLite supports PyTorch-like device management method.

.. code-block:: python

    # Backload the engine to GPU
    engine.cuda()  # Move the engine to GPU, same as `engine.to("cuda")`
    engine.cuda("weights")  # Only move the weights to GPU, useful for vllm.
    engine.cuda("kv_cache")  # Only move the KV cache to GPU, useful for vllm.

    engine.cpu()  # Offload the engine to CPU, same as `engine.to("cpu")`
    engine.meta()  # Release the weights completely, same as `engine.to("meta")`

Together with the ``p2p_weight_sync`` interface, RLite allows efficient weight synchronization between different engines.

.. code-block:: python

    # Suppose engine_A is on GPU, and we want to sync the weights to vllm engine_B
    engine_B.cuda("weights")
    engine_A.p2p_weight_sync(engine_B)
    engine_A.cpu()
    engine_B.cuda("kv_cache")

Resource Management Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RLite support specifying the colocation or non-colocation relationship between engines. The interface is very simple, just passing ``colocate_with=target_engine`` or ``not_colocate_with=target_engine`` when calling the ``build`` method of the engine.

.. code-block:: python

    engine.build(
        ...,
        colocate_with=engine_A,
        not_colocate_with=[engine_B, engine_C],
    )

This interface, together with the ``acceptable_node_ips`` argument of ``build`` method, can be used to implement the placement of engines on the same set of GPUs or on different (maybe heterogeneous) machines.

.. code-block:: python

    engine.build(
        ...,
        acceptable_node_ips=["192.168.1.1", "192.168.1.2"],
    )


Code Structure
--------------

.. code-block:: text

   rlite
    ├── inference
    ├── interface
    ├── nn
    ├── resman
    ├── third_party
    ├── train
    ├── utils
    ├── __init__.py
    └── __version__.py

The ``interface`` directory defines all core interfaces, while the ``nn`` directory contains all core modules. These two directories form the heart of RLite, and users can refer to the code in these directories to quickly understand RLite's programming model.

The ``train`` directory contains the specific implementation of the training engine, and the ``inference`` directory contains the specific implementation of the inference engine.

The ``resman`` directory contains the resource management module.

Limitations
-----------

Due to engineering capacity and development time constraints, RLite currently has several limitations.

- RLite does not yet support multi-dimensional parallel distributed training engines like Megatron-core or advanced parallel strategies such as expert parallelism, which makes RLite suboptimal in performance when training/inferencing ultra-large-scale models. Thankfully, our current Ray-based training engine interface design can accommodate these efficient distributed training engines, and we plan to add support for these advanced parallel strategies in future versions.
