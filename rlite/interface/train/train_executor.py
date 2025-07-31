import abc

import ray

from rlite.interface.base_executor import BaseExecutor
from rlite.train.utils import prepare_parallel_args, prepare_parallel_kwargs
from rlite.utils.registry import Registry


class BaseTrainExecutor(BaseExecutor, abc.ABC):
    def run_workers(self, method_name: str, *args, **kwargs) -> list[ray.ObjectRef]:
        """Run a method on all workers.

        This function allows calling the methods defined in the module. The
        args and kwargs will be parsed according to its NeedParallel type.
        """
        # Magic part.
        args_list = prepare_parallel_args(args, unwrap=True, dp_size=len(self.workers))
        kwargs_list = prepare_parallel_kwargs(kwargs, unwrap=True, dp_size=len(self.workers))
        if hasattr(self.workers[0], method_name):
            futures = [
                getattr(worker, method_name).remote(*args, **kwargs)
                for worker, args, kwargs in zip(self.workers, args_list, kwargs_list)
            ]
        else:
            futures = [
                worker.getattr.remote(method_name, *args, **kwargs)
                for worker, args, kwargs in zip(self.workers, args_list, kwargs_list)
            ]
        return futures

    def train(self) -> list[ray.ObjectRef]:
        """Switch to training mode."""
        return [worker.train.remote() for worker in self.workers]

    def eval(self) -> list[ray.ObjectRef]:
        """Switch to evaluation mode."""
        return [worker.eval.remote() for worker in self.workers]

    def is_training(self) -> list[bool]:
        return [worker.is_training() for worker in self.workers]

    def save(self, checkpoint_path: str, *args, **kwargs) -> list[ray.ObjectRef]:
        return [worker.save.remote(checkpoint_path, *args, **kwargs) for worker in self.workers]

    def load(self, checkpoint_path: str, *args, **kwargs) -> list[ray.ObjectRef]:
        return [worker.load.remote(checkpoint_path, *args, **kwargs) for worker in self.workers]


TRAIN_EXECUTOR_REGISTRY = Registry("Train Executors")
