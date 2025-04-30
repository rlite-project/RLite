from __future__ import annotations

import json
import os
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Any, Generator, Optional, Union

import cloudpickle
import pynvml
import torch
import torch.distributed
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.distributed_c10d import (
    Backend,
    GroupMember,
    PrefixStore,
    ProcessGroup,
    Store,
    _new_process_group_helper,
    _shutdown_backend,
    _unregister_all_process_groups,
    _unregister_process_group,
    _update_default_pg,
    _world,
    default_pg_timeout,
    rendezvous
)
from tqdm import tqdm


class CUDAIPCHandle:
    def __init__(
        self,
        tensor_type: type,
        size: tuple,
        stride: tuple,
        offset: int,
        storage_type: type,
        dtype: torch.dtype,
        device: torch.device,
        handle: bytes,
        storage_size_bytes: bytes,
        storage_offset_bytes: bytes,
        requires_grad: bool,
        ref_counter_handle: bytes,
        ref_counter_offset: bytes,
        event_handle: bytes,
        event_sync_required: bool,
    ):
        self.tensor_type = tensor_type
        self.size = size
        self.stride = stride
        self.offset = offset
        self.storage_type = storage_type
        self.dtype = dtype
        self.device = device
        self.handle = handle
        self.storage_size_bytes = storage_size_bytes
        self.storage_offset_bytes = storage_offset_bytes
        self.requires_grad = requires_grad
        self.ref_counter_handle = ref_counter_handle
        self.ref_counter_offset = ref_counter_offset
        self.event_handle = event_handle
        self.event_sync_required = event_sync_required

    def rebuild(self) -> torch.Tensor:
        # NOTE: Rebuild within the same process is not thread-safe and will
        #       likely crash (segfault core dump).
        if self.handle is None:
            raise ValueError("Storage handle is None")
        torch.cuda._lazy_init()
        storage = self.storage_type._new_shared_cuda(
            self.device,
            self.handle,
            self.storage_size_bytes,
            self.storage_offset_bytes,
            self.ref_counter_handle,
            self.ref_counter_offset,
            self.event_handle,
            self.event_sync_required,
        )
        _storage = (
            storage
            if isinstance(storage, torch.UntypedStorage)
            else storage._untyped_storage
        )

        self.storage_type._release_ipc_counter(
            self.ref_counter_handle,
            self.ref_counter_offset,
            device=self.device
        )

        t = torch._utils._rebuild_tensor(
            torch.storage.TypedStorage(wrap_storage=_storage, dtype=self.dtype, _internal=True),
            self.offset,
            self.size,
            self.stride,
        )

        if self.tensor_type == torch.nn.parameter.Parameter:
            # It is crucial for integer tensors to receive
            # the requires_grad=False as an argument in the constructor
            t = torch.nn.parameter.Parameter(t, requires_grad=self.requires_grad)
        else:
            t.requires_grad = self.requires_grad

        return t.clone()

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> CUDAIPCHandle:
        if not tensor.is_cuda:
            raise ValueError("Tensor must be on CUDA device to use CUDAIPC")
        tensor = tensor.clone().share_memory_()
        storage = tensor._typed_storage()
        (
            device,
            handle,
            storage_size_bytes,
            storage_offset_bytes,
            ref_counter_handle,
            ref_counter_offset,
            event_handle,
            event_sync_required,
        ) = storage._share_cuda_()
        tensor_offset = tensor.storage_offset()
        storage._release_ipc_counter(ref_counter_handle, ref_counter_offset, device=device)
        return CUDAIPCHandle(
            tensor_type=type(tensor),
            size=tensor.size(),
            stride=tensor.stride(),
            # tensor offset in its storage
            offset=tensor_offset,
            storage_type=type(storage),
            dtype=tensor.dtype,
            device=device,
            # identifier which CUDA allocation is the storage in.
            handle=handle,
            # size(in bytes) of the storage
            storage_size_bytes=storage_size_bytes,
            # offset(in bytes) of the storage in the CUDA allocation
            storage_offset_bytes=storage_offset_bytes,
            requires_grad=tensor.requires_grad,
            ref_counter_handle=ref_counter_handle,
            ref_counter_offset=ref_counter_offset,
            event_handle=event_handle,
            event_sync_required=event_sync_required,
        )


def get_device_uuid(device_id: int = 0) -> str:
    pynvml.nvmlInit()
    try:
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            device_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            if device_ids == [""]:
                raise RuntimeError(
                    "CUDA_VISIBLE_DEVICES is set to empty string,"
                    " which means GPU support is disabled."
                )
            physical_device_id = int(device_ids[device_id])
        else:
            physical_device_id = device_id

        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
        return pynvml.nvmlDeviceGetUUID(handle)
    finally:
        pynvml.nvmlShutdown()


def load_safetensors_generator(
    safetensors_path,
    device="cpu",
    use_tqdm: bool = True,
    fetch_data: bool = True,
    dtype: torch.dtype = torch.float32,
):
    path = Path(safetensors_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {safetensors_path}")

    if path.is_dir():
        paths = list(path.glob("*.safetensors"))
        if len(paths) == 0:
            raise FileNotFoundError(f"No safetensors files found in directory: {safetensors_path}")
    else:
        paths = [path]

    if use_tqdm:
        paths = tqdm(paths, desc="Loading safetensors")

    for safetensors_path in paths:
        with safe_open(safetensors_path, framework="pt", device=device) as f:
            for key in f.keys():
                if fetch_data:
                    yield key, f.get_tensor(key).to(dtype)
                else:
                    yield key, None


def save_safetensors(
    state_dict: dict | Generator,
    path: str,
    min_block_size_in_GB: int = 5,
    max_block_size_in_GB: int = 20,
    total_size: int | None = None,
    total_len: int | None = None,
    is_io_worker: bool = True,
):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if total_size is None and isinstance(state_dict, dict):
        total_size = 0
        for param in state_dict.values():
            total_size += param.numel() * param.element_size()

    if total_size is not None:
        size_per_block = total_size // 4 + 1
        size_per_block = min(size_per_block, max_block_size_in_GB * 1024**3)
        size_per_block = max(size_per_block, min_block_size_in_GB * 1024**3)
    else:
        from loguru import logger

        logger.warning(
            f"Total size is not provided, using min block size: {min_block_size_in_GB}GB."
        )
        size_per_block = min_block_size_in_GB * 1024**3

    if total_len is None and isinstance(state_dict, dict):
        total_len = len(state_dict)

    if isinstance(state_dict, dict):
        state_dict = state_dict.items()

    partial_state_dict = {}
    num_saved_files = 0
    partial_state_dict_size = 0
    safetensors_path = path / "model-1.safetensors"
    meta_info = {"metadata": {"total_size": total_size}, "weight_map": {}}

    for name, param in tqdm(
        state_dict,
        total=total_len,
        desc="Saving weights",
        disable=not is_io_worker,
    ):
        partial_state_dict[name] = param
        partial_state_dict_size += param.numel() * param.element_size()
        meta_info["weight_map"][name] = safetensors_path.name
        if partial_state_dict_size > size_per_block:
            if is_io_worker:
                save_file(partial_state_dict, safetensors_path, metadata={"format": "pt"})
            num_saved_files += 1
            safetensors_path = path / f"model-{num_saved_files + 1}.safetensors"

            partial_state_dict = {}
            partial_state_dict_size = 0

    if len(partial_state_dict) > 0 and is_io_worker:
        save_file(partial_state_dict, safetensors_path, metadata={"format": "pt"})

    if is_io_worker:
        with open(path / "model.safetensors.index.json", "w") as f:
            json.dump(meta_info, f)


def sync_weights(
    source_model,
    target_model,
    source_model_reside: str = "cpu",
    target_model_reside: str = "cuda"
):
    """Synchronize the weights from the train model to the inference model.

    This should happen when train model is on GPU and inference model is on CPU
    or Meta.
    """
    # TODO (zhanghan):
    # we can further reduce the GPU memory usage by lazy loading the weights.
    # That is, during iterating the generator, we gradually materialize the
    # inference model's weights on GPU. In the meantime, we can free the
    # weights from the train model. This allows that at any time, only 1x of
    # the model size is on GPU. Currently, we always keep 2x of the model size
    # on GPU.
    torch.cuda.empty_cache()
    source_model.clear_cache()
    target_model.clear_cache()

    if source_model.device != "cuda":
        source_model.cuda(init_cache_engine=False)
    if target_model.device != "cuda":
        target_model.cuda(init_cache_engine=False)
    for model in (source_model, target_model):
        if hasattr(model, "is_cache_engine_active") and model.is_cache_engine_active():
            model.free_cache_engine()

    if source_model.is_colocate_with(target_model):
        # P2P weight sync via CUDA IPC
        source_model.sync_weights_cudaipc(target_model)
    else:
        # All-gather, shared CPU memory, and broadcast
        target_model.load_state_dict(source_model.state_dict_generator())

    torch.cuda.empty_cache()
    source_model.to(source_model_reside)
    target_model.to(target_model_reside)


class SerializableModule(torch.nn.Module):
    def __getstate__(self):
        return cloudpickle.dumps(self.__dict__)

    def __setstate__(self, state):
        self.__dict__.update(cloudpickle.loads(state))


# Copy from pytorch to allow creating multiple main groups.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py
def init_process_group(
    backend: Union[str, Backend] = None,
    init_method: Optional[str] = None,
    timeout: Optional[timedelta] = None,
    world_size: int = -1,
    rank: int = -1,
    store: Optional[Store] = None,
    group_name: str = None,
    pg_options: Optional[Any] = None,
):
    assert (store is None) or (init_method is None), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)

    pg_options_param_name = "backend_options" if str(torch.__version__) >= "2.6" else "pg_options"
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        **{pg_options_param_name: pg_options},
        timeout=timeout,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg


def destroy_process_group(group: Optional[ProcessGroup] = None):
    """
    Destroy a given process group, and deinitialize the distributed package.

    Args:
        group (ProcessGroup, optional): The process group to be destroyed, if
                                        group.WORLD is given, all process
                                        groups including the default one will
                                        be destroyed.
    """
    global _world

    if group == GroupMember.NON_GROUP_MEMBER:
        return

    if group is None:
        pg = GroupMember.WORLD
    else:
        pg = group

    assert pg is not None
    if _world.pg_map.get(pg, None) is None:
        raise ValueError("Invalid process group specified")

    # When users register Python onCompletion hooks, those hooks will run on a
    # different thread than the main thread. Today, the ProcessGroup dtor does
    # wait for that thread. However, the dtor might finish after the Python
    # Interpreter exits. After that grabbing the GIL for the Python hook will
    # crash. We can either revive the interpreter when running hooks or keep
    # the main one alive until all works and hooks are done. The current
    # implementation does the latter. Therefore, we explicitly call
    # _wait_for_pending_works() here to wait for the pending hooks to finish.
    if pg.name().lower() == "nccl" and pg._has_hooks():
        pg._wait_for_pending_works()

    if group is None or group == GroupMember.WORLD:
        # shutdown all backends in the order of pg names. shutting down in
        # order because ncclCommAbort() was a 'collective' call in some
        # versions of NCCL.
        for pg_to_shutdown in sorted(
            _world.pg_names,
            key=lambda x: _world.pg_names[x],
            reverse=True
        ):
            _shutdown_backend(pg_to_shutdown)

        _update_default_pg(None)
        _world.pg_map.clear()
        _world.pg_names.clear()
        _world.pg_group_ranks.clear()
        _world.pg_backend_config.clear()
        _world.pg_to_tag.clear()
        _world.tags_to_pg.clear()
        _world.pg_coalesce_state.clear()
        _world.pg_default_device.clear()
        _unregister_all_process_groups()

        # when process group doesn't have an explicit name (only WORLD
        # (default) process group can have an explicit name), we use global
        # _world.group_count to generate the name. We need to reset the counter
        # on destruction to allow consistent value to be generated when we
        # re-create process groups after some trainers recover from failure.
        #
        # We only reset this when WORLD is being destroyed because if this
        # process group is in good state, we aren't dealing with failures.
        _world.group_count = 0
    else:
        _shutdown_backend(pg)
        del _world.pg_map[pg]
        del _world.pg_names[pg]
        del _world.pg_group_ranks[pg]
        del _world.pg_backend_config[pg]
        if pg in _world.pg_default_device:
            del _world.pg_default_device[pg]
        if pg in _world.pg_coalesce_state.keys():
            warnings.warn(
                "Some coalesced collectives haven't been launched when "
                "ProcessGroup is destroyed. They will be cleaned."
            )
            del _world.pg_coalesce_state[pg]

        tag = _world.pg_to_tag.get(pg)
        del _world.pg_to_tag[pg]
        if tag is not None:
            try:
                _world.tags_to_pg[tag].remove(pg)
                if tag.startswith("ptd:"):
                    _world.tags_to_pg[""].remove(pg)
            except Exception:
                pass
        _unregister_process_group(pg.group_name)
