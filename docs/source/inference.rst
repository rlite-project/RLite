Inference with RLite
====================

RLite's inference engine provides a ``generate`` interface similar to HuggingFace and vLLM. Currently supported backends include:

- vLLM

Inference with Language Models
------------------------------

Initializing the inference engine only requires specifying the model path and backend engine. In the ``build`` function, users need to specify the model's parallelism strategy, including:

- ``tensor_parallel_size``: Size of tensor parallelism
- ``pipeline_parallel_size``: Size of pipeline parallelism
- ``data_parallel_size``: Size of data parallelism, defaults to -1, which means using as many DP Executors as possible

Other backend parameters can also be specified through the ``build`` function.

.. code-block:: python

    engine = rlite.InferenceEngine(
        model_path="path/to/model",
        backend="vllm",
    )
    engine.build(
        tensor_parallel_size=4,
        gpu_memory_utilization=0.9,
    )

Prompts and generation parameters can be passed to the ``generate`` function, where generation parameters support vLLM's ``SamplingParams`` or can be directly passed as a dictionary. The generate function also supports ``tqdm`` progress bars. By default, all prompts are processed in parallel, with the backend engine randomly distributing these prompts to different executors.

.. code-block:: python

    engine.generate(
        ["Hello, world!"] * 3,
        use_tqdm=True,
        tqdm_desc="Generating random samples",
    )

Inference with Vision-Language Models
-------------------------------------

Inference with Vision-Language Models is similar to language models, requiring only the model path and backend engine to be specified. The input `prompt` is simply a dictionary containing the raw image and text.

.. code-block:: python

    engine.generate(
        [
            {
                "prompt": "What is the image about?",
                "multi_modal_data": {
                    "image": PIL.Image.open("path/to/image.jpg")
                }
            }
            for _ in range(8)
        ],
        sampling_params={
            "temperature": 0.7,
            "seed": 42,
        },
        use_tqdm=True,
        tqdm_desc="VLM generating",
    )
