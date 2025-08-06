from __future__ import annotations

from vllm import RequestOutput


class IndexedGenerationHistory:
    def __init__(
        self,
        index: int,
        inputs: list[str] | None = None,
        outputs: list[RequestOutput] | None = None
    ):
        self.index = index
        self.inputs = inputs if inputs is not None else []
        self.outputs = outputs if outputs is not None else []


class StopGeneration:
    """A special class to signal the end of generation."""
