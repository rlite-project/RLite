from rlite.nn.fsdp import FsdpTrainModule
from rlite.resman import ResourceBundle
from rlite.train.fsdp.worker import FsdpWorker
from rlite.train.interface.train_executor import (
    TRAIN_EXECUTOR_REGISTRY,
    BaseTrainExecutor
)
from rlite.utils.misc import set_random_seed


@TRAIN_EXECUTOR_REGISTRY.register([
    "fsdp", "fsdp1", "FSDP", "FSDP1", "Fsdp", "Fsdp1", "FsdpTrainExecutor"
])
class FsdpTrainExecutor(BaseTrainExecutor):
    def __init__(
        self,
        *,
        dp_rank: int,
        module: FsdpTrainModule,
        bundles: list[ResourceBundle],
        tensor_parallel_size: int,
        data_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        seed: int = 42,
        **fsdp_kwargs
    ):
        assert isinstance(module, FsdpTrainModule), type(module)
        self.dp_rank = dp_rank
        self.module = module
        self.bundles = bundles
        self.tensor_parallel_size = tensor_parallel_size
        self.data_parallel_size = data_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.seed = seed
        self.fsdp_kwargs = fsdp_kwargs

        self._workers: list[FsdpWorker] = []

    @property
    def workers(self):
        return self._workers

    def build_workers(self):
        self._workers = self._build_workers(
            module=self.module,
            dp_rank=self.dp_rank,
            bundles=self.bundles,
            data_parallel_size=self.data_parallel_size,
            tensor_parallel_size=self.tensor_parallel_size,
            pipeline_parallel_size=self.pipeline_parallel_size,
            seed=self.seed,
            **self.fsdp_kwargs
        )
        return [worker.ready.remote() for worker in self._workers]

    def _build_workers(
        self,
        module: FsdpTrainModule,
        dp_rank: int,
        bundles: list[ResourceBundle],
        data_parallel_size: int,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        seed: int,
        replicate_master_addrs: list[str] | None = None,
        replicate_ports: list[int] | None = None,
        **fsdp_kwargs
    ):
        set_random_seed(seed)

        workers = []
        assert len(bundles) == tensor_parallel_size * pipeline_parallel_size, (
            f"The number of bundles ({len(bundles)=}) "
            "should be equal to the product of "
            f"tensor_parallel_size ({tensor_parallel_size=}) "
            f"and pipeline_parallel_size ({pipeline_parallel_size=})"
        )

        assert pipeline_parallel_size == 1, "Pipeline parallel is not supported yet!"
        assert tensor_parallel_size > 1, "FSDP requires tensor parallel size > 1"

        # Prepare the env variables
        shard_master_addr = bundles[0].resource_manager.get_ip_addr(bundles[0])
        shard_port = bundles[0].resource_manager.get_free_port(bundles[0])
        world_size = tensor_parallel_size * pipeline_parallel_size * data_parallel_size
        if data_parallel_size > 1:
            node_rank, num_nodes = dp_rank, data_parallel_size
        else:
            node_rank, num_nodes = 0, 1

        for rank in range(tensor_parallel_size * pipeline_parallel_size):
            bundle = bundles[rank]
            env_vars = {
                "RANK": rank,
                "WORLD_SIZE": world_size,
                "NODE_RANK": node_rank,
                "NUM_NODES": num_nodes,
                "SHARD_MASTER_ADDR": shard_master_addr,
                "SHARD_MASTER_PORT": shard_port,
            }
            if replicate_master_addrs is not None:
                env_vars["REPLICATE_MASTER_ADDR"] = replicate_master_addrs[rank]
                env_vars["REPLICATE_MASTER_PORT"] = replicate_ports[rank]
            env_vars = {k: str(v) for k, v in env_vars.items()}

            worker = bundle.remote(
                FsdpWorker,
                runtime_env={"env_vars": env_vars},
            ).remote(module=module, **fsdp_kwargs)
            workers.append(worker)

        return workers
