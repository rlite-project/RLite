from __future__ import annotations

from functools import cached_property, wraps
from typing import Generator

import ray
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from rlite.interface import BaseWorker
from rlite.interface.train import BaseTrainWorker
from rlite.nn import Fsdp2TrainModule
from rlite.train.utils.fsdp import offload_optimizer, reload_optimizer
from rlite.utils.checkpoint import load_safetensors_generator, save_safetensors
from rlite.utils.distributed import CUDAIPCHandle


def abort_if_not_cuda(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.get_device() != "cuda":
            raise RuntimeError(
                f"Currently the model is on {self.get_device()}, you need to call "
                "`cuda()` first to save the checkpoint."
            )
        return func(self, *args, **kwargs)
    return wrapper


def capture_forward_error(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "got mixed torch.Tensor and DTensor" in str(e):
                raise RuntimeError(
                    "FSDP2 does not support calling `forward` directly. That is, "
                    "calling `self.forward(...)` is expected to fail. Please call "
                    "`self(...)` instead."
                )
            raise e
    return wrapper


class Fsdp2Worker(BaseTrainWorker):
    def __init__(
        self,
        module: Fsdp2TrainModule,
        mesh_shape: tuple[int, ...],
        rank_in_shard_group: int,
        shard_group_world_size: int,
        **kwargs,
    ):
        self.module = module.float()
        self.rank_in_shard_group = rank_in_shard_group
        self.shard_group_world_size = shard_group_world_size
        self._init_model(self.module, mesh_shape)

    def __getattr__(self, name):
        if name == "model":
            return object.__getattribute__(self, "model")
        try:
            return super().__getattr__(name)
        except AttributeError:
            # MAGIC: Hijack the getattr to support any customized method.
            model = object.__getattribute__(self, "model")
            try:
                return nn.Module.__getattr__(model, name)
            except AttributeError:
                if hasattr(self.module, name):
                    attr = getattr(self.module, name)
                    if callable(attr):
                        return capture_forward_error(attr.__get__(model))
                raise

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    @property
    def training(self) -> bool:
        return self.model.training

    def cpu(self):
        self.model.cpu()
        if hasattr(self, "optimizer"):
            offload_optimizer(self.model.optimizer)
        self.release_memory_cache()

    def cuda(self):
        self.model.cuda()
        if hasattr(self, "optimizer"):
            reload_optimizer(self.model.optimizer)

    def meta(self):
        raise NotImplementedError("Currently not implemented, coming soon.")

    def get_device(self) -> str:
        return next(self.model.parameters()).device.type

    @abort_if_not_cuda
    def sync_weights_to(self, target_worker: ray.ActorHandle[BaseWorker]):
        for key, param in self._named_parameters.items():
            postprocess_fn = self.module.postprocess_output_named_parameters.__get__(self.model)
            key, param = postprocess_fn(key, param, self._named_parameters)
            ipc_handle = CUDAIPCHandle.from_tensor(param)
            future = target_worker.update_weight.remote(key, ipc_handle)
            ray.get(future)

        ray.get(target_worker.update_weight.remote("END"))

    @abort_if_not_cuda
    def update_weight(self, key: str, weight: torch.Tensor | CUDAIPCHandle | None = None):
        if key == "END":
            return
        if isinstance(weight, CUDAIPCHandle):
            weight = weight.rebuild()
        preprocess_fn = self.module.preprocess_input_named_parameters.__get__(self.model)
        key, weight = preprocess_fn(key, weight)
        self._named_parameters[key]._local_tensor.data.copy_(weight)

    @abort_if_not_cuda
    def state_dict(self) -> dict:
        local_state_dict = self.model.state_dict()
        postprocess_fn = self.module.postprocess_output_named_parameters.__get__(self.model)
        full_state_dict = dict([
            postprocess_fn(key, value, local_state_dict)
            for key, value in local_state_dict.items()
        ])
        if dist.get_rank() == 0:
            return full_state_dict

    @abort_if_not_cuda
    def load_state_dict(self, state_dict: dict | Generator):
        # TODO: Support quantized state_dict loading. Directly loading a
        #       quantized state dict from the safe tensors will run into
        #       KeyError, e.g. the GPT-oss model.

        if isinstance(state_dict, dict):
            state_dict = state_dict.items()

        exist_keys = set()

        for key, value in state_dict:

            exist_keys.add(key)

            if key not in self._named_parameters:
                continue

            # Apply the preprocess function
            preprocess_fn = self.module.preprocess_input_named_parameters.__get__(self.model)
            key, value = preprocess_fn(key, value)

            # Chunk and load the weights
            chunked_value = torch.chunk(
                value, self.shard_group_world_size, dim=0
            )[self.rank_in_shard_group]
            self._named_parameters[key]._local_tensor.data.copy_(chunked_value)

        for key in self._named_parameters:
            if key not in exist_keys:
                raise KeyError(f"Key '{key}' not found in the state_dict.")

    @abort_if_not_cuda
    def save(self, checkpoint_path: str):
        def _param_iterator():
            post_process_fn = self.module.postprocess_output_named_parameters.__get__(self.model)
            for name, param in self._named_parameters.items():
                yield post_process_fn(name, param, self._named_parameters)
        save_safetensors(
            _param_iterator(),
            checkpoint_path,
            total_len=len(self._named_parameters),
            is_io_worker=self.rank_in_shard_group == 0
        )

    @abort_if_not_cuda
    def load(self, checkpoint_path: str):
        state_dict_iterator = load_safetensors_generator(
            checkpoint_path,
            device="cuda",
            use_tqdm=self.rank_in_shard_group == 0,
            dtype=torch.bfloat16
        )
        self.load_state_dict(state_dict_iterator)

    # Private methods

    def _init_model(self, module: Fsdp2TrainModule, mesh_shape: tuple[int, ...]):
        device_mesh = self._init_device_mesh(mesh_shape)

        named_buffers = {
            name: buffer.cpu().clone()
            for name, buffer in module.named_buffers()
        }

        self.model = self._init_fsdp(module, device_mesh)

        for name, buffer in self.model.named_buffers():
            buffer.data.copy_(named_buffers[name])

        # Rebind the attributes
        for key, value in module.__dict__.items():
            if not key.startswith("_") and not hasattr(self.model, key):
                setattr(self.model, key, value)

        mat_src = module.materialization_source()
        if isinstance(mat_src, str):
            self.load(mat_src)
        elif isinstance(mat_src, dict):
            raise NotImplementedError("Currently not implemented, coming soon.")
        else:
            raise ValueError(f"Unknown materialization source: {mat_src}")

        # Call the on_weights_materialized hook if it exists
        if hasattr(self.module, "on_weights_materialized"):
            self.module.on_weights_materialized.__get__(self.model)()

    def _init_device_mesh(self, mesh_shape: tuple[int, ...]):
        from torch.distributed.device_mesh import init_device_mesh

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(0)
        device_mesh = init_device_mesh("cuda", mesh_shape, mesh_dim_names=("replicate", "shard"))
        return device_mesh

    def _init_fsdp(self, module: Fsdp2TrainModule, device_mesh: DeviceMesh):
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
        for sub_module in module.modules():
            if sub_module.__class__ in module.non_split_modules:
                fully_shard(sub_module, mesh=device_mesh, mp_policy=mp_policy)
        fully_shard(module, mesh=device_mesh, mp_policy=mp_policy)
        module.to_empty(device="cuda")
        module.init_weights()
        return module

    @cached_property
    def _named_parameters(self):
        return dict(self.model.named_parameters())

    def __del__(self):
        if dist.is_initialized():
            dist.destroy_process_group()
