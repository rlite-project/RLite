from __future__ import annotations

import abc

from ray.util.queue import Queue
from vllm import RequestOutput, SamplingParams

from rlite.interface import BaseExecutor
from rlite.utils.registry import Registry


class BaseInferenceExecutor(BaseExecutor, abc.ABC):
    @abc.abstractmethod
    def generate(
        self,
        input_queue: Queue,
        output_queue: Queue,
        sampling_params: dict | SamplingParams | None = None,
        **kwargs
    ) -> list[RequestOutput | None]:
        """Generate the output for the given prompts."""
        futures = [
            worker.generate.remote(
                input_queue=input_queue,
                output_queue=output_queue,
                sampling_params=sampling_params,
            )
            for worker in self.workers
        ]
        return futures

    def device(self) -> str:
        futures = [worker.get_device.remote() for worker in self.workers]
        return futures


INFERENCE_EXECUTOR_REGISTRY = Registry("Inference Executors")
