from __future__ import annotations

import ray

from rlite.interface.base_engine import BaseEngine, abort_if_no_executor
from rlite.nn import BaseTrainModule
from rlite.train.interface import TRAIN_EXECUTOR_REGISTRY, BaseTrainExecutor
from rlite.train.utils import prepare_parallel_args, prepare_parallel_kwargs


class RliteTrainEngine(BaseEngine):
    def __init__(
        self,
        module: BaseTrainModule,
        *,
        executor: BaseTrainExecutor | str = "fsdp2",
        **kwargs
    ):
        super().__init__(**kwargs)

        assert isinstance(module, BaseTrainModule)
        assert next(module.parameters()).is_meta, "Use meta model to initialize!"
        self._module = module

        # Call the init_hook of the module
        self._module.init_hook()

        if isinstance(executor, str):
            self._executor_cls = TRAIN_EXECUTOR_REGISTRY[executor]
        else:
            self._executor_cls = executor

    @property
    def executor_cls(self) -> type[BaseTrainExecutor]:
        return self._executor_cls

    @abort_if_no_executor
    def state_dict(self) -> dict:
        return ray.get(self.executors[0].state_dict())

    @abort_if_no_executor
    def load_state_dict(self, state_dict: dict):
        # NOTE: This function is expected to be slow. Use CUDAIPC + colocate
        #       to speed up the transfer.
        futures = sum([
            executor.load_state_dict(state_dict)
            for executor in self.executors
        ], [])
        return ray.get(futures)

    @abort_if_no_executor
    def train(self):
        """Switch to training mode."""
        futures = sum([executor.train() for executor in self.executors], [])
        return ray.get(futures)

    @abort_if_no_executor
    def eval(self):
        """Switch to evaluation mode."""
        futures = sum([executor.eval() for executor in self.executors], [])
        return ray.get(futures)

    @property
    @abort_if_no_executor
    def training(self) -> bool:
        is_training = ray.get(sum([executor.is_training() for executor in self.executors]))
        if all(is_training):
            return True
        elif not any(is_training):
            return False
        else:
            raise RuntimeError("Mixed training and evaluation executors!")

    def __getattr__(self, attr_name: str) -> any:
        try:
            return object.__getattribute__(self, attr_name)
        except AttributeError:

            # Abort if no executor is available.
            if self.executors is None or len(self.executors) == 0:
                raise AttributeError(f"No attribute {attr_name} found in the engine.")

            # Magic part.
            # This function allows dynamically calling methods defined in the
            # module on all executors.
            def run_executors(*args, **kwargs):
                # Note that we don't unwrap the args and kwargs here, as the
                # executor can also employ data parallel, e.g. FSDP.
                args_list = prepare_parallel_args(
                    args,
                    unwrap=False,
                    dp_size=len(self.executors)
                )
                kwargs_list = prepare_parallel_kwargs(
                    kwargs,
                    unwrap=False,
                    dp_size=len(self.executors)
                )
                futures = [
                    executor.run_workers(attr_name, *args_list[dp_rank], **kwargs_list[dp_rank])
                    for dp_rank, executor in enumerate(self.executors)
                ]
                return [ray.get(future) for future in futures]

            return run_executors

    def pre_build_executors_hook(
        self,
        *,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        data_parallel_size: int,
        **kwargs,
    ) -> dict:
        from rlite.train.fsdp2.executor import Fsdp2TrainExecutor
        from rlite.train.fsdp.executor import FsdpTrainExecutor

        if self.executor_cls is FsdpTrainExecutor and data_parallel_size > 1:
            # Prepare replicate group master address and ports.
            num_gpus_per_dp_group = tensor_parallel_size * pipeline_parallel_size
            replicate_master_addrs = [
                self._resource_manager.get_ip_addr(self._resource_bundles[i])
                for i in range(num_gpus_per_dp_group)
            ]
            replicate_ports = [
                self._resource_manager.get_free_port(self._resource_bundles[i])
                for i in range(num_gpus_per_dp_group)
            ]
            return {
                "module": self._module,
                "replicate_master_addrs": replicate_master_addrs,
                "replicate_ports": replicate_ports,
            }

        elif self.executor_cls is Fsdp2TrainExecutor:
            master_addr = self._resource_manager.get_ip_addr(self._resource_bundles[0])
            master_port = self._resource_manager.get_free_port(self._resource_bundles[0])
            mesh_shape = (data_parallel_size, tensor_parallel_size * pipeline_parallel_size)
            return {
                "module": self._module,
                "master_addr": master_addr,
                "master_port": master_port,
                "mesh_shape": mesh_shape,
            }

        return {"module": self._module}

    def save(self, checkpoint_path: str, *args, **kwargs):
        ray.get(self.executors[0].save(checkpoint_path, *args, **kwargs))

    def load(self, checkpoint_path: str, *args, **kwargs):
        futures = sum([
            executor.load(checkpoint_path, *args, **kwargs)
            for executor in self.executors
        ], [])
        return ray.get(futures)
