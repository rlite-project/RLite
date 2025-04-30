from __future__ import annotations

import os
from functools import reduce
from pathlib import Path
from typing import Literal

import ray
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    MixedPrecision,
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictType
)

from rlite.interface import BaseWorker
from rlite.nn.fsdp import FsdpTrainModule
from rlite.train.interface import BaseTrainWorker
from rlite.train.utils.fsdp import (
    iter_fsdp1_state_dict,
    offload_model,
    offload_optimizer,
    reload_model,
    reload_optimizer
)
from rlite.utils.checkpoint import save_safetensors
from rlite.utils.distributed import (
    CUDAIPCHandle,
    destroy_process_group,
    init_process_group
)


class FsdpWorker(BaseTrainWorker):
    def __init__(self, module: FsdpTrainModule, *, device_id: int = 0):
        self._filter_warnings()
        self.model = self._init_model(module, device_id)

    def __getattr__(self, name):
        model = object.__getattribute__(self, "model")
        if name == "model":
            return model
        try:
            return super().__getattr__(name)
        except AttributeError:
            # NOTE: Here we hijack the getattr of the FSDP to make sure that
            #       the getattr of the model is called with `self` being the
            #       FSDP instance instead of the _fsdp_wrapped_module.
            try:
                return nn.Module.__getattr__(model, name)
            except AttributeError:
                if hasattr(model._fsdp_wrapped_module.__class__, name):
                    attr = getattr(model._fsdp_wrapped_module.__class__, name)
                    if callable(attr):
                        return attr.__get__(model)
                return getattr(model._fsdp_wrapped_module, name)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def cpu(self):
        offload_model(self.model)
        if hasattr(self, "optimizer"):
            offload_optimizer(self.model.optimizer)
        self.release_memory_cache()

    def cuda(self, *args, **kwargs):
        reload_model(self.model)
        if hasattr(self.model, "optimizer"):
            reload_optimizer(self.model.optimizer)

    def meta(self):
        raise NotImplementedError("Currently not implemented, coming soon.")

    def sync_weights_to(self, target_worker: ray.ActorHandle[BaseWorker]):
        for key, value in iter_fsdp1_state_dict(self.model):
            ipc_handle = CUDAIPCHandle.from_tensor(value)
            future = target_worker.update_weight.remote(key, ipc_handle)
            ray.get(future)

        ray.get(target_worker.update_weight.remote("END"))

    def update_weight(
        self,
        key: str,
        weight: torch.Tensor | CUDAIPCHandle | None = None,
    ):
        if key == "END":
            with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT):
                self.model.load_state_dict(self._weight_load_state_dict)
                delattr(self, "_weight_load_state_dict")
            return

        if not hasattr(self, "_weight_load_state_dict"):
            with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT):
                self._weight_load_state_dict = self.model.state_dict()

        if isinstance(weight, CUDAIPCHandle):
            weight = weight.rebuild()
        if weight.device.type == "meta":
            weight = torch.empty_like(weight, device="cuda")
        if weight.device.type == "cpu":
            weight = weight.to("cuda", non_blocking=True)

        sharded_value = self._weight_load_state_dict[key]
        shard_dim = sharded_value._sharding_spec.dim
        v = torch.chunk(weight.cuda(), self.local_world_size, dim=shard_dim)[self.local_rank]
        sharded_value.local_shards()[0].tensor.copy_(v)

    def state_dict(self) -> dict:
        state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, state_dict_config):
            return self.model.state_dict()

    def load_state_dict(self, state_dict: dict):
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
            self.model.load_state_dict(state_dict)

    @property
    def training(self) -> bool:
        return self.model.training

    def get_device(self) -> str:
        if not hasattr(self, "model"):
            raise RuntimeError(
                "FSDP worker is not initialized! You probably need to call "
                "`build_worker_groups` first."
            )
        return next(self.model.parameters()).device.type

    # See example here:
    # https://github.com/pytorch/pytorch/blob/main/torch/distributed/checkpoint/examples/fsdp_checkpoint_example.py  # noqa
    def save(
        self,
        checkpoint_path: str,
        type: Literal["distcp", "safetensors"] = "safetensors"
    ):
        if self.get_device() != "cuda":
            raise RuntimeError(
                f"Currently the model is on {self.get_device()}, you need to call "
                "`cuda()` first to save the checkpoint."
            )

        if type == "distcp":
            self._save_distcp(checkpoint_path)
        elif type == "safetensors":
            self._save_safetensors(checkpoint_path)

    def load(self, checkpoint_id: str, type: Literal["distcp", "safetensors", "auto"] = "auto"):
        if self.get_device() != "cuda":
            raise RuntimeError(
                f"Currently the model is on {self.get_device()}, you need to call "
                "`cuda()` first to load the checkpoint."
            )

        if type == "auto":
            # Determine the type of the checkpoint
            ckpt_path = Path(checkpoint_id)
            if any(p.is_file() for p in ckpt_path.glob("*.safetensors")):
                type = "safetensors"
            elif any(p.is_file() for p in ckpt_path.glob("*.distcp")):
                type = "distcp"

        valid_types = ["distcp", "safetensors", "auto"]
        if type not in valid_types:
            raise ValueError(f"Unknown checkpoint type: {type}, must be one of {valid_types}")

        if type == "safetensors":
            self._load_safetensors(checkpoint_id)
        elif type == "distcp":
            self._load_distcp(checkpoint_id)

    def _save_safetensors(self, checkpoint_id: str):
        with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT):
            sharded_state_dict = self.model.state_dict()

        # Get the total number of parameters and the state dict size in Bytes
        total_len = len(sharded_state_dict)
        total_size, element_size = 0, None
        for value in sharded_state_dict.values():
            if element_size is None:
                element_size = torch.tensor([], dtype=value.dtype).element_size()
            numel = reduce(lambda x, y: x * y, value.shape)
            total_size += numel * element_size

        # Memory-efficiently save the state dict
        state_dict_iterator = iter_fsdp1_state_dict(sharded_state_dict=sharded_state_dict)
        save_safetensors(
            state_dict_iterator,
            checkpoint_id,
            total_size=total_size,
            total_len=total_len,
            is_io_worker=self.local_rank == 0,
        )

    def _save_distcp(self, checkpoint_id: str):
        with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT):
            state_dict = {
                "model": self.model.state_dict(),
            }
            if hasattr(self, "optim"):
                state_dict["optim"] = FSDP.optim_state_dict(self.model, self.optimizer)
            dcp.save(
                state_dict,
                storage_writer=dcp.FileSystemWriter(checkpoint_id),
                process_group=self.shard_group
            )

    def _load_safetensors(self, checkpoint_id: str, model=None):
        from rlite.utils.checkpoint import load_safetensors_generator

        model = model or self.model
        current_model_device = next(model.parameters()).device.type

        if current_model_device != "cuda":
            raise RuntimeError("Currently the model is on CPU, so we cannot load the checkpoint.")

        with FSDP.state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            state_dict_config=ShardedStateDictConfig(offload_to_cpu=False)
        ):
            sharded_state_dict = model.state_dict()
            dtype = next(iter(sharded_state_dict.values())).dtype

        state_dict_iterator = load_safetensors_generator(
            checkpoint_id,
            device="cuda",
            use_tqdm=self.local_rank == 0,
            fetch_data=self.local_rank == 0,
            dtype=dtype
        )

        for key, value in state_dict_iterator:
            sharded_value = sharded_state_dict[key]
            value_shape = list(sharded_value.shape)
            value_shape[sharded_value._sharding_spec.dim] //= self.local_world_size
            recv_tensor = torch.empty(torch.Size(value_shape), device="cuda", dtype=dtype)
            if self.local_rank == 0:
                tensor_list = list(
                    torch.chunk(
                        value,
                        self.local_world_size,
                        dim=sharded_value._sharding_spec.dim
                    )
                )
            else:
                tensor_list = None
            dist.scatter(recv_tensor, tensor_list, src=0, group=self.shard_group)
            sharded_value.local_shards()[0].tensor.copy_(recv_tensor)

        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            model.load_state_dict(sharded_state_dict)

    def _load_distcp(self, checkpoint_id: str):
        with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT):
            state_dict = {
                "model": self.model.state_dict(),
            }
            dcp.load(
                state_dict,
                storage_reader=dcp.FileSystemReader(checkpoint_id),
                process_group=self.shard_group
            )
            self.model.load_state_dict(state_dict["model"])
            if hasattr(self, "optimizer"):
                from torch.distributed.checkpoint.optimizer import (
                    load_sharded_optimizer_state_dict
                )

                FSDP.optim_state_dict(self.model, self.optimizer, state_dict["optimizer"])
                optim_state = load_sharded_optimizer_state_dict(
                    model_state_dict=state_dict["model"],
                    optimizer_key="optim",
                    storage_reader=dcp.FileSystemReader(checkpoint_id),
                )
                flattened_osd = FSDP.optim_state_dict_to_load(
                    self.model, self.optimizer, optim_state["optim"]
                )
                self.optimizer.load_state_dict(flattened_osd)

    def _init_optimizer(
        self,
        optimizer_cls: type | str | None = None,
        learning_rate: float = 1e-5,
        optimizer_init_kwargs: dict | None = None,
    ):
        if isinstance(optimizer_cls, str):
            optimizer_cls = getattr(torch.optim, optimizer_cls)
        elif optimizer_cls is None:
            optimizer_cls = torch.optim.AdamW

        optimizer_init_kwargs = optimizer_init_kwargs or {}
        optimizer_init_kwargs["lr"] = learning_rate
        optimizer = optimizer_cls(self.model.parameters(), **optimizer_init_kwargs)
        return optimizer

    def _init_model(self, module: FsdpTrainModule, device_id: int = 0):
        # 1. Initialize process groups
        self.shard_group, self.replicate_group = self._init_process_groups()

        # 2. Determine the sharding strategy and process group
        if self.replicate_group is None:
            self.sharding_strategy = ShardingStrategy.FULL_SHARD
            process_group = self.shard_group
        else:
            self.sharding_strategy = ShardingStrategy.HYBRID_SHARD
            process_group = (self.shard_group, self.replicate_group)

        # 3. Materialize the model weights
        mat_src = module.materialization_source()
        if isinstance(mat_src, str):
            folder = Path(mat_src)
            wrap_policy = module.get_fsdp_wrap_policy()
            if len(list(folder.glob("*.safetensors"))) > 0:
                # TODO (zhanghan): support other checkpoint formats
                model = self._init_model_by_checkpoint_hf(
                    module, device_id, process_group, wrap_policy
                )
            else:
                model = self._init_model_by_checkpoint_distcp(
                    module, device_id, process_group, wrap_policy
                )
        elif isinstance(mat_src, dict):
            # For random initialization and train from scratch
            raise NotImplementedError("Currently not implemented, coming soon.")

        # 4. Set communication attributes for outside world
        model.shard_rank = model.module.shard_rank = self.local_rank
        model.shard_world_size = model.module.shard_world_size = self.local_world_size
        model.replicate_rank = model.module.replicate_rank = self.node_rank
        model.replicate_world_size = model.module.replicate_world_size = self.num_nodes

        # 4. Call hooks
        if hasattr(model, "on_weights_materialized"):
            model.on_weights_materialized()

        # 5. Explicitly set the model to training mode to avoid some bugs
        model.train()

        return model

    def _init_model_by_checkpoint_hf(self, module, device_id, process_group, wrap_policy):
        model = self._init_fsdp(
            module,
            device_id=device_id,
            process_group=process_group,
            sharding_strategy=self.sharding_strategy,
            auto_wrap_policy=wrap_policy,
        )

        self._load_safetensors(module.get_checkpoint_path(), model)

        return model

    def _init_model_by_checkpoint_distcp(self, module, device_id, process_group, wrap_policy):
        model = self._init_fsdp(
            module,
            device_id=device_id,
            process_group=process_group,
            sharding_strategy=self.sharding_strategy,
            auto_wrap_policy=wrap_policy,
        )
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            state_dict = {"model": model.state_dict()}
            dcp.load(
                state_dict,
                storage_reader=dcp.FileSystemReader(model.get_checkpoint_path()),
                process_group=self.shard_group,
            )
            model.load_state_dict(state_dict["model"])
            model = model.to(next(module.parameters()).dtype)
        return model

    def _init_fsdp(self, module, **kwargs):
        from rlite.train.utils.fsdp import param_init_preserve_buffers

        kwargs = dict(
            {
                "mixed_precision": MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    buffer_dtype=torch.float32
                ),
                "param_init_fn": param_init_preserve_buffers,
                "forward_prefetch": True,
                "backward_prefetch": "BACKWARD_PRE",
                "use_orig_params": False,
                "device_id": "cuda",
            },
            **kwargs,
        )
        # For FSDP training model, float32 weights are required, otherwise the
        # performance can be problematic.
        return FSDP(module.float(), **kwargs)

    def _init_process_groups(self):
        # Collect distributed settings
        self.rank = int(os.getenv("RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))
        self.node_rank = int(os.getenv("NODE_RANK", 0))
        self.num_nodes = int(os.getenv("NUM_NODES", 1))
        self.shard_master_addr = os.getenv("SHARD_MASTER_ADDR")
        self.shard_master_port = int(os.getenv("SHARD_MASTER_PORT"))

        self.local_world_size = self.world_size // self.num_nodes
        self.local_rank = self.rank % self.local_world_size

        # Initialize process groups
        shard_init_method = f"tcp://{self.shard_master_addr}:{self.shard_master_port}"
        conflict_free_shard_name = f"fsdp_shard_{self.shard_master_addr}_{self.shard_master_port}"
        dist.init_process_group(
            rank=self.local_rank,
            world_size=self.local_world_size,
            backend="nccl",
            group_name=conflict_free_shard_name,
            init_method=shard_init_method,
        )
        shard_group = dist.group.WORLD
        if self.num_nodes > 1:
            replicate_master_addr = os.getenv("REPLICATE_MASTER_ADDR")
            replicate_master_port = int(os.getenv("REPLICATE_MASTER_PORT"))
            replicate_init_method = f"tcp://{replicate_master_addr}:{replicate_master_port}"
            conflict_free_replicate_name = (
                f"fsdp_replicate_{replicate_master_addr}_{replicate_master_port}"
            )
            replicate_group = init_process_group(
                rank=self.node_rank,
                world_size=self.num_nodes,
                backend="nccl",
                group_name=conflict_free_replicate_name,
                init_method=replicate_init_method,
            )
            return shard_group, replicate_group
        return shard_group, None

    def _filter_warnings(self):
        import warnings

        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*_get_pg_default_device.*"
        )

    def __del__(self):
        if hasattr(self, "shard_group") and self.shard_group is not None:
            dist.destroy_process_group(self.shard_group)
        if hasattr(self, "replicate_group") and self.replicate_group is not None:
            destroy_process_group(self.replicate_group)

    # ----- Generator style state dict utilities ----- #

    # NOTE (zhanghan): these functions are very useful if we want to sync
    #                  weights more efficiently without CUDAIPC, while still
    #                  keeping the memory footprint low. These functions are
    #                  based on a fake generator implementation based on
    #                  message passing. The ray generator does not work, as it
    #                  does not pause the actor when the generator yields. It
    #                  will run the loop and save the yielded values in a
    #                  buffer. This can cause deaklock or messed up CUDA IPC.

    def _setup_state_dict_generator(self):
        with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT):
            self._state_dict_iterator = iter(self.model.state_dict().items())

    def _generate_next_state(self):
        try:
            key, value = next(self.state_dict_iterator)
            gathered_value = torch.empty(
                value.shape,
                device=torch.device("cuda"),
                dtype=value.dtype,
            )
            value.gather(out=gathered_value if dist.get_rank() == 0 else None)
            if dist.get_rank() == 0:
                return key, gathered_value.cpu()
        except StopIteration:
            delattr(self, "_state_dict_iterator")
            self.clear_cache()
            return "END", None
        except AttributeError:
            return

    def _setup_state_dict_loader(self):
        """Load a full state dict."""
        with FSDP.state_dict_type(
            self.model,
            StateDictType.SHARDED_STATE_DICT,
            state_dict_config=ShardedStateDictConfig(offload_to_cpu=False)
        ):
            self._state_dict_loader = self.model.state_dict()

    def _load_next_state(
        self,
        key: str,
        value: torch.Tensor | None = None,
        broadcast_weight: bool = False,
    ):
        if key == "END":
            with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT):
                self.model.load_state_dict(self._state_dict_loader)
                delattr(self, "_state_dict_loader")
                return

        # Reconstruct the weights
        if isinstance(value, CUDAIPCHandle):
            value = value.rebuild()
        if value.device.type == "meta":
            value = torch.empty_like(value, device="cuda")
        if value.device.type == "cpu":
            value = value.to("cuda", non_blocking=True)

        if broadcast_weight:
            dist.broadcast(value, src=0, group=self.shard_group)

        sharded_value = self._state_dict_loader[key]
        shard_dim = sharded_value._sharding_spec.dim
        v = torch.chunk(value, self.local_world_size, dim=shard_dim)[self.local_rank].cuda()
        sharded_value.local_shards()[0].tensor.copy_(v)
