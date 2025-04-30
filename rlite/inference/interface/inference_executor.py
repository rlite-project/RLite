from __future__ import annotations

import abc

from vllm import RequestOutput, SamplingParams

from rlite.interface import BaseExecutor
from rlite.utils.registry import Registry


class BaseInferenceExecutor(BaseExecutor, abc.ABC):
    @abc.abstractmethod
    def generate(
        self,
        prompts: list[str],
        sampling_params: dict | SamplingParams | None = None,
        use_tqdm: bool = True,
        tqdm_desc: str = "Processed prompts",
        return_none: bool = False,
    ) -> list[RequestOutput | None]:
        """Generate the output for the given prompts."""
        futures = [
            worker.generate.remote(
                prompts=prompts,
                sampling_params=sampling_params,
                use_tqdm=use_tqdm,
                tqdm_desc=tqdm_desc,
                return_none=(rank != 0)
            )
            for rank, worker in enumerate(self.workers)
        ]
        return futures


INFERENCE_EXECUTOR_REGISTRY = Registry("Inference Executors")
