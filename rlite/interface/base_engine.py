from __future__ import annotations

import abc
from functools import wraps

import ray

from rlite.interface.base_executor import BaseExecutor
from rlite.resman import ResourceConsumer, ResourceManager


def abort_if_no_executor(func: callable):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.executors is None or len(self.executors) == 0:
            raise RuntimeError("Executor is not built")
        return func(self, *args, **kwargs)
    return wrapper


class BaseEngine(ResourceConsumer, abc.ABC):
    # ----- Interfaces ----- #

    @property
    @abc.abstractmethod
    def executor_cls(self) -> type[BaseExecutor]:
        raise NotImplementedError()

    # ----- Shared methods and properties ----- #

    def __init__(self, resource_manager: ResourceManager | None = None):
        ResourceConsumer.__init__(self, resource_manager)
        self._executors: list[BaseExecutor] = []

    @property
    def executors(self) -> list:
        return self._executors

    # Build methods

    def build(
        self,
        *,
        tensor_parallel_size: int,
        pipeline_parallel_size: int = 1,
        data_parallel_size: int = -1,
        allow_colocate: bool = True,
        colocate_with: ResourceConsumer | None = None,
        not_colocate_with: list[ResourceConsumer] | ResourceConsumer | None = None,
        ignore_rebuild_error: bool = False,
        **executor_kwargs
    ):
        if not self.is_registered():
            raise RuntimeError("Current engine is not registered in any resource manager!")

        if self.is_resource_allocated() and not ignore_rebuild_error:
            raise RuntimeError("Executors are already built")

        if data_parallel_size == -1:
            data_parallel_size = self._resource_manager.total_gpus // (
                tensor_parallel_size * pipeline_parallel_size
            )
            assert data_parallel_size > 0, "Not enough GPUs available!"

        # Ask for resources from resource manager
        num_gpu_request = tensor_parallel_size * pipeline_parallel_size * data_parallel_size
        self._resource_manager.allocate_resources(
            consumer=self,
            num_gpus=num_gpu_request,
            allow_colocate=allow_colocate,
            colocate_with=colocate_with,
            not_colocate_with=not_colocate_with,
        )

        # Allow some additional parameters to be added. this is useful, e.g.,
        # for FSDP, as different dp groups can also sync gradients (hybrid
        # sharding).
        additional_params = self.pre_build_executors_hook(
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            data_parallel_size=data_parallel_size,
            allow_colocate=allow_colocate,
            colocate_with=colocate_with,
            not_colocate_with=not_colocate_with,
            ignore_rebuild_error=ignore_rebuild_error,
            **executor_kwargs
        )
        executor_kwargs.update(additional_params)

        # Assign resources to each executors (each executor is a dp group)
        executors = []
        for dp_rank in range(data_parallel_size):
            executor = self.build_executor(
                dp_rank=dp_rank,
                data_parallel_size=data_parallel_size,
                tensor_parallel_size=tensor_parallel_size,
                pipeline_parallel_size=pipeline_parallel_size,
                **executor_kwargs
            )
            executors.append(executor)
        self._executors = executors

        # Build workers
        futures = sum([executor.build_workers() for executor in self.executors], [])
        ray.get(futures)

    def build_executor(
        self,
        dp_rank: int,
        data_parallel_size: int,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        **executor_kwargs
    ):
        num_gpus_per_dp = tensor_parallel_size * pipeline_parallel_size
        bundle_index_offset = dp_rank * num_gpus_per_dp
        bundles = self._resource_bundles[bundle_index_offset:bundle_index_offset + num_gpus_per_dp]
        executor = self.executor_cls(
            bundles=bundles,
            dp_rank=dp_rank,
            data_parallel_size=data_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            **executor_kwargs
        )
        return executor

    def pre_build_executors_hook(self, **kwargs) -> dict:
        return {}

    # State dict methods

    @abort_if_no_executor
    def load_state_dict(self, state_dict: dict):
        futures = sum([executor.load_state_dict(state_dict) for executor in self.executors], [])
        ray.get(futures)

    @abort_if_no_executor
    def state_dict(self) -> dict:
        # By default, the state dict is on CPU and gathered to rank 0
        return ray.get(self.executors[0].state_dict())[0]

    @abort_if_no_executor
    def p2p_weight_sync(self, another: BaseEngine):
        assert self.is_colocate_with(another), "Must be colocated to use p2p weight sync!"

        my_workers = sum([executor.workers for executor in self.executors], [])
        another_workers = sum([executor.workers for executor in another.executors], [])

        futures = [
            worker.sync_weights_to.remote(another_worker)
            for worker, another_worker in zip(my_workers, another_workers)
        ]
        ray.get(futures)

    # Device methods

    @abort_if_no_executor
    def cpu(self, *args, **kwargs):
        futures = sum([executor.cpu(*args, **kwargs) for executor in self.executors], [])
        ray.get(futures)

    @abort_if_no_executor
    def meta(self, *args, **kwargs):
        futures = sum([executor.meta(*args, **kwargs) for executor in self.executors], [])
        ray.get(futures)

    @abort_if_no_executor
    def cuda(self, *args, **kwargs):
        futures = sum([executor.cuda(*args, **kwargs) for executor in self.executors], [])
        ray.get(futures)

    @abort_if_no_executor
    def to(self, device: str, *args, **kwargs):
        futures = sum([executor.to(device, *args, **kwargs) for executor in self.executors], [])
        ray.get(futures)

    @abort_if_no_executor
    def device(self) -> list[str] | str:
        futures = sum([executor.device() for executor in self.executors], [])
        devices = ray.get(futures)
        if all(device == devices[0] for device in devices):
            return devices[0]
        else:
            return devices

    @abort_if_no_executor
    def release_memory_cache(self, cpu: bool = False, gpu: bool = True):
        futures = sum([executor.release_memory_cache(cpu, gpu) for executor in self.executors], [])
        ray.get(futures)
