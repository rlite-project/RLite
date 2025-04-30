from __future__ import annotations

import abc

import ray

from rlite.interface.base_worker import BaseWorker


class BaseExecutor(abc.ABC):
    # ----- Interfaces ----- #

    @property
    @abc.abstractmethod
    def workers(self) -> list[BaseWorker]:
        raise NotImplementedError()

    @abc.abstractmethod
    def build_workers(self) -> list[ray.ObjectRef]:
        raise NotImplementedError()

    # ----- Shared methods ----- #

    def load_state_dict(self, state_dict: dict) -> list[ray.ObjectRef]:
        futures = [worker.load_state_dict.remote(state_dict) for worker in self.workers]
        return futures

    def state_dict(self) -> list[ray.ObjectRef]:
        futures = [worker.state_dict.remote() for worker in self.workers]
        return futures

    def cpu(self, *args, **kwargs) -> list[ray.ObjectRef]:
        futures = [worker.cpu.remote(*args, **kwargs) for worker in self.workers]
        return futures

    def meta(self, *args, **kwargs) -> list[ray.ObjectRef]:
        futures = [worker.meta.remote(*args, **kwargs) for worker in self.workers]
        return futures

    def cuda(self, *args, **kwargs) -> list[ray.ObjectRef]:
        futures = [worker.cuda.remote(*args, **kwargs) for worker in self.workers]
        return futures

    def to(self, device: str, *args, **kwargs) -> list[ray.ObjectRef]:
        """Move the workers to the given device."""
        assert device in ["cpu", "cuda", "meta"]
        if device == "cpu":
            return self.cpu(*args, **kwargs)
        elif device == "cuda":
            return self.cuda(*args, **kwargs)
        elif device == "meta":
            return self.meta(*args, **kwargs)

    def release_memory_cache(self, cpu: bool = False, gpu: bool = True) -> list[ray.ObjectRef]:
        """Release the memory cache."""
        futures = [worker.release_memory_cache.remote(cpu, gpu) for worker in self.workers]
        return futures

    def get_device(self) -> list[ray.ObjectRef]:
        futures = [worker.get_device.remote() for worker in self.workers]
        return futures
