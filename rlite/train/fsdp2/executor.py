import abc

from rlite.interface.train.train_executor import (
    TRAIN_EXECUTOR_REGISTRY,
    BaseTrainExecutor
)
from rlite.nn import Fsdp2TrainModule
from rlite.resman import ResourceBundle
from rlite.train.fsdp2.worker import Fsdp2Worker
from rlite.utils.misc import set_random_seed


@TRAIN_EXECUTOR_REGISTRY.register([
    "fsdp2", "FSDP2", "Fsdp2", "Fsdp2TrainExecutor"
])
class Fsdp2TrainExecutor(BaseTrainExecutor, abc.ABC):
    def __init__(
        self,
        *,
        dp_rank: int,
        module: Fsdp2TrainModule,
        bundles: list[ResourceBundle],
        tensor_parallel_size: int,
        data_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        seed: int = 42,
        master_addr: str,
        master_port: int,
        mesh_shape: tuple[int, ...],
        **fsdp_kwargs
    ):
        # assert isinstance(module, Fsdp2TrainModule), type(module)
        self.dp_rank = dp_rank
        self.module = module
        self.bundles = bundles
        self.tensor_parallel_size = tensor_parallel_size
        self.data_parallel_size = data_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.seed = seed
        self.fsdp_kwargs = fsdp_kwargs
        self.master_addr = master_addr
        self.master_port = master_port
        self.mesh_shape = mesh_shape

        self._workers: list[Fsdp2Worker] = []

    @property
    def workers(self):
        return self._workers

    def build_workers(self):
        self._workers = self._build_workers(
            module=self.module,
            bundles=self.bundles,
            master_addr=self.master_addr,
            master_port=self.master_port,
            mesh_shape=self.mesh_shape,
            **self.fsdp_kwargs
        )
        return [worker.ready.remote() for worker in self._workers]

    def _build_workers(
        self,
        module: Fsdp2TrainModule,
        bundles: list[ResourceBundle],
        master_addr: str,
        master_port: int,
        mesh_shape: tuple[int, ...],
        **fsdp_kwargs
    ):
        set_random_seed(self.seed)
        assert len(bundles) == self.tensor_parallel_size * self.pipeline_parallel_size, (
            f"The number of bundles ({len(bundles)=}) "
            "should be equal to the product of "
            f"tensor_parallel_size ({self.tensor_parallel_size=}) "
            f"and pipeline_parallel_size ({self.pipeline_parallel_size=})"
        )

        assert self.pipeline_parallel_size == 1, "Pipeline parallel is not supported yet!"
        assert self.tensor_parallel_size > 1, "FSDP2 requires tensor parallel size > 1"

        workers = []
        for rank in range(self.tensor_parallel_size * self.pipeline_parallel_size):
            bundle = bundles[rank]
            global_rank = (
                self.dp_rank * self.tensor_parallel_size * self.pipeline_parallel_size + rank
            )
            global_world_size = (
                self.data_parallel_size * self.tensor_parallel_size * self.pipeline_parallel_size
            )
            worker = bundle.remote(
                Fsdp2Worker,
                runtime_env={"env_vars": {
                    "RANK": str(global_rank),
                    "WORLD_SIZE": str(global_world_size),
                    "MASTER_ADDR": master_addr,
                    "MASTER_PORT": str(master_port),
                }}
            ).remote(
                module=module,
                mesh_shape=mesh_shape,
                rank_in_shard_group=rank,
                shard_group_world_size=self.tensor_parallel_size * self.pipeline_parallel_size,
                **fsdp_kwargs
            )
            workers.append(worker)
        return workers
