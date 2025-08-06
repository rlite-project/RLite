import ray
from ray.util.queue import Queue
from vllm import SamplingParams

from rlite.inference.vllm.worker import VllmWorker
from rlite.interface.inference.inference_executor import (
    INFERENCE_EXECUTOR_REGISTRY,
    BaseInferenceExecutor
)
from rlite.resman import ResourceBundle
from rlite.utils.misc import set_random_seed


@INFERENCE_EXECUTOR_REGISTRY.register(names=["vllm", "vLLM", "VLLM", "VllmInferenceExecutor"])
class VllmInferenceExecutor(BaseInferenceExecutor):
    def __init__(
        self,
        *,
        model_path: str,
        bundles: list[ResourceBundle],
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        seed: int = 42,
        # env related arguments
        use_v1_engine: bool = True,
        enable_vllm_logs: bool = False,
        **vllm_kwargs
    ):
        self.model_path = model_path
        self.bundles = bundles
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.seed = seed
        self.use_v1_engine = use_v1_engine
        self.enable_vllm_logs = enable_vllm_logs
        self.vllm_kwargs = vllm_kwargs

        self._workers: list[VllmWorker] = []

    def build_workers(self):
        self._workers: list[VllmWorker] = self._build_workers(
            model_path=self.model_path,
            bundles=self.bundles,
            tensor_parallel_size=self.tensor_parallel_size,
            pipeline_parallel_size=self.pipeline_parallel_size,
            seed=self.seed,
            use_sleep_mode=True,
            use_v1_engine=self.use_v1_engine,
            enable_vllm_logs=self.enable_vllm_logs,
            **self.vllm_kwargs
        )
        return [worker.ready.remote() for worker in self._workers]

    @property
    def workers(self) -> list[VllmWorker]:
        return self._workers

    def generate(
        self,
        input_queue: Queue,
        output_queue: Queue,
        sampling_params: dict | SamplingParams | None = None,
    ) -> list[ray.ObjectRef]:
        return [
            worker.generate.remote(
                input_queue,
                output_queue,
                sampling_params,
                is_io_worker=(rank == 0)
            )
            for rank, worker in enumerate(self.workers)
        ]

    def p2p_weight_sync(self, another: BaseInferenceExecutor):
        raise NotImplementedError(
            "Currently VllmInferenceExecutor does not support p2p weight sync!"
        )

    def state_dict(self):
        raise NotImplementedError("Currently VllmInferenceExecutor does not support state_dict()")

    def load_state_dict(self, state_dict: dict):
        futures = [
            worker.load_state_dict.remote(state_dict)
            for worker in self.workers
        ]
        return futures

    def _build_workers(
        self,
        model_path: str,
        bundles: list[ResourceBundle],
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        seed: int,
        use_v1_engine: bool,
        enable_vllm_logs: bool,
        **vllm_kwargs
    ) -> list[VllmWorker]:
        set_random_seed(seed)
        workers = []
        assert len(bundles) == tensor_parallel_size * pipeline_parallel_size, (
            f"The number of bundles ({len(bundles)=}) "
            "should be equal to the product of "
            f"tensor_parallel_size ({tensor_parallel_size=}) "
            f"and pipeline_parallel_size ({pipeline_parallel_size=})"
        )

        # TODO (zhanghan): handle pipeline parallel later
        assert pipeline_parallel_size == 1, "Pipeline parallel is not supported yet!"

        master_addr = bundles[0].resource_manager.get_ip_addr(bundles[0])
        # NOTE: there are chances that the port is already used by other
        #       processes, as we don't lock the port.
        master_port = bundles[0].resource_manager.get_free_port(bundles[0])

        for rank in range(tensor_parallel_size * pipeline_parallel_size):
            bundle = bundles[rank]

            env_vars = {
                "RANK": str(rank),
                "WORLD_SIZE": len(bundles),
                "LOCAL_RANK": "0",
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": master_port,
                "VLLM_USE_V1": int(use_v1_engine),
                "VLLM_CONFIGURE_LOGGING": int(enable_vllm_logs),
            }
            env_vars = {k: str(v) for k, v in env_vars.items()}

            worker = bundle.remote(
                VllmWorker,
                runtime_env={"env_vars": env_vars}
            ).remote(
                model_path=model_path,
                seed=seed,
                tensor_parallel_size=tensor_parallel_size,
                pipeline_parallel_size=pipeline_parallel_size,
                **vllm_kwargs
            )
            workers.append(worker)

        return workers
