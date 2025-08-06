from __future__ import annotations

import random

import ray
from loguru import logger
from ray.util.queue import Empty, Queue
from vllm import RequestOutput, SamplingParams

import rlite
from rlite.inference.utils import IndexedGenerationHistory, StopGeneration
from rlite.interface.base_engine import BaseEngine, abort_if_no_executor
from rlite.interface.inference.inference_executor import (
    INFERENCE_EXECUTOR_REGISTRY,
    BaseInferenceExecutor
)
from rlite.nn.inference_module import BaseInferenceModule
from rlite.resman import ResourceBundle
from rlite.utils.need_parallel import NeedParallel


def default_post_generation_hook(
    generation: IndexedGenerationHistory,
    input_queue: Queue,
    output_queue: Queue
):
    """By default, we finish generation after this round."""
    # NOTE: In fact, we can omit this default hook and let the monitor do the
    #       post-processing. This default hook keeps the code cleaner at a cost
    #       of more ray remote calls.
    output_queue.put(generation)


class PostGenerationHookMonitor:
    def __init__(
        self,
        hook: callable,
        input_queue: Queue,
        intermediate_queue: Queue,
        output_queue: Queue,
        resource_bundles: list[ResourceBundle]
    ):
        self.hook = hook
        self.input_queue = input_queue
        self.intermediate_queue = intermediate_queue
        self.output_queue = output_queue
        self.resource_bundles = resource_bundles

    def run(self):
        pending_futures = []

        while True:
            if pending_futures:
                ready, pending_futures = ray.wait(
                    pending_futures,
                    num_returns=len(pending_futures),
                    timeout=0  # Non-blocking wait
                )

                for future in ready:
                    ray.get(future)

            try:
                item = self.intermediate_queue.get(timeout=0.1)

                if isinstance(item, StopGeneration):
                    if pending_futures:
                        ray.get(pending_futures)
                    return

                # TODO: figure out a cleverer way to assign the resouce bundle
                bundle = random.choice(self.resource_bundles)

                future = bundle.remote(self.hook, num_cpus=0.1).remote(
                    generation=item,
                    input_queue=self.input_queue,
                    output_queue=self.output_queue
                )
                pending_futures.append(future)

            except Empty:
                continue


class OutputCollector:
    def __init__(
        self,
        input_queue: Queue,
        intermediate_queue: Queue,
        output_queue: Queue,
        total: int,
        num_executors: int,
        use_tqdm: bool = True,
        tqdm_desc: str = "Processed prompts"
    ):
        self.input_queue = input_queue
        self.intermediate_queue = intermediate_queue
        self.output_queue = output_queue
        self.total = total
        self.num_executors = num_executors

        if use_tqdm:
            from tqdm import tqdm
            self.pbar = tqdm(total=total, desc=tqdm_desc)
        else:
            self.pbar = None

    def run(self) -> list[IndexedGenerationHistory]:
        outputs: list[IndexedGenerationHistory] = []
        while len(outputs) < self.total:
            outputs.append(self.output_queue.get())
            if self.pbar:
                self.pbar.update(1)
                self.pbar.refresh()
        sorted_outputs = sorted(outputs, key=lambda x: x.index)

        # Stop the generation
        for _ in range(self.num_executors):
            self.input_queue.put(StopGeneration())

        # Stop the monitor
        self.intermediate_queue.put(StopGeneration())

        if self.pbar:
            self.pbar.close()

        return [output.outputs for output in sorted_outputs]


class RliteInferenceEngine(BaseEngine):
    def __init__(
        self,
        module_or_path: str | BaseInferenceModule,
        executor: BaseInferenceExecutor | str = "vllm",
        **kwargs
    ):
        super().__init__(**kwargs)

        if isinstance(module_or_path, BaseInferenceModule):
            self._module_path = module_or_path.get_checkpoint_path()
        else:
            self._module_path = module_or_path

        if isinstance(executor, str):
            self._executor_cls = INFERENCE_EXECUTOR_REGISTRY[executor]
        else:
            self._executor_cls = executor

    @property
    def executor_cls(self) -> type[BaseInferenceExecutor]:
        return self._executor_cls

    @abort_if_no_executor
    def generate(
        self,
        prompts: list[str],
        sampling_params: dict | SamplingParams | None = None,
        *,
        post_generation_hook: callable | None = None,
        use_tqdm: bool = True,
        tqdm_desc: str = "Processed prompts"
    ) -> list[RequestOutput | list[RequestOutput]]:
        """
        Generate responses for the given prompts.

        The engines will be fully asynchronous. After they finish one sample,
        they will check if there are more samples to process. If so, they will
        continue processing the next sample. If not, they will wait for new
        samples to arrive.

        The `post_generation_hook` should have the same signature as
        `default_post_generation_hook`, which is called after each generation.
        """

        if isinstance(sampling_params, dict):
            sampling_params = SamplingParams(**sampling_params)

        if isinstance(prompts, NeedParallel):
            prompts = prompts.unwrap()
            logger.info(
                "NOTE: rlite by default uses data parallel during "
                "inference, and the sequence parallel is handled by the "
                "backend inference engine."
            )

        if post_generation_hook is None:
            post_generation_hook = default_post_generation_hook

        input_queue, intermediate_queue, output_queue = Queue(), Queue(), Queue()
        for index, prompt in enumerate(prompts):
            input_queue.put(
                IndexedGenerationHistory(index=index, inputs=[prompt])
            )

        futures = []

        # NOTE: The inference executors are not aware of the intermediate
        #       queue. It just execute the generation and deliver the results.
        #       The intermediate queue is monitored by another actor, which
        #       will call the post generation hook. This design allows fully
        #       asynchronous execution of the generation and potential
        #       expensive hooks.
        for executor in self.executors:
            futures.extend(
                executor.generate(
                    input_queue=input_queue,
                    output_queue=intermediate_queue,
                    sampling_params=sampling_params,
                )
            )

        # Post-generation hook monitor
        monitor = ray.remote(PostGenerationHookMonitor).remote(
            hook=post_generation_hook,
            input_queue=input_queue,
            intermediate_queue=intermediate_queue,
            output_queue=output_queue,
            resource_bundles=rlite.resman.RESMAN.resource_bundles
        )
        futures.append(monitor.run.remote())

        # Output collector
        collector = ray.remote(OutputCollector).remote(
            input_queue=input_queue,
            intermediate_queue=intermediate_queue,
            output_queue=output_queue,
            total=len(prompts),
            num_executors=len(self.executors),
            use_tqdm=use_tqdm,
            tqdm_desc=tqdm_desc
        )
        futures.append(collector.run.remote())

        res = ray.get(futures)  # Ensure all tasks are completed

        return res[-1]  # Return the collected outputs

    def pre_build_executors_hook(self, **kwargs) -> dict:
        return {"model_path": self._module_path}
