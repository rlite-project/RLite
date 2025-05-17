Inference with RLite
====================


Inference with Language Models
------------------------------



Inference with Vision-Language Models
-------------------------------------


Progress Bar
------------

The ``generate`` method of the inference engine supports ``tqdm`` progress bar.

.. code-block:: python

    engine.generate(
        ["Hello, world!"] * 3,
        use_tqdm=True,
        tqdm_desc="Generating random samples",
    )
