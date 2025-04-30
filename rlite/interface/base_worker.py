from __future__ import annotations

import abc

import ray
import torch

from rlite.utils.distributed import CUDAIPCHandle


class BaseWorker(abc.ABC):
    # ----- Interfaces ----- #

    @abc.abstractmethod
    def update_weight(self, key: str, value: torch.Tensor | CUDAIPCHandle | None):
        """Update a single weight of the worker.

        This is useful for sync the weights between colocated workers. The key
        can be "END" to indicate the end of the sync (useful for FSDP).
        """

    @abc.abstractmethod
    def sync_weights_to(self, worker: ray.ActorHandle[BaseWorker]):
        """Sync the weights to another worker."""

    @abc.abstractmethod
    def load_state_dict(self, state_dict: dict):
        """Load the state dict of the worker."""

    @abc.abstractmethod
    def state_dict(self) -> dict:
        """Get the state dict of the worker."""

    @abc.abstractmethod
    def cpu(self, *args, **kwargs):
        """Move the worker to the CPU."""

    @abc.abstractmethod
    def meta(self, *args, **kwargs):
        """Release all the weights (not including buffers)."""

    @abc.abstractmethod
    def cuda(self, *args, **kwargs):
        """Move the worker to the GPU."""

    @abc.abstractmethod
    def get_device(self) -> str:
        """Get the device of the worker."""

    # ----- Shared methods ----- #

    def release_memory_cache(self, cpu: bool = False, gpu: bool = True):
        if cpu:
            try:
                torch._C._host_emptyCache()
            except AttributeError:
                from loguru import logger

                logger.warning(
                    "torch._C._host_emptyCache() only available in Pytorch >=2.5"
                )
        if gpu:
            torch.cuda.empty_cache()

    def ready(self):
        """Check if the worker is ready."""
        return True
