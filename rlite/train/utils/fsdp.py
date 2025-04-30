from __future__ import annotations

from collections import defaultdict

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp._runtime_utils import _lazy_init


def ensure_weights_retied(param_init_fn, model: torch.nn.Module, device: torch.cuda.device):
    """Handles `transformers` models with tied weights."""

    _tied_names = getattr(model, "_tied_weights_keys", None)
    if not _tied_names:
        # if no tied names just passthrough
        return param_init_fn

    # get map of parameter instances to params.
    # - needed for replacement later
    _tied_params = {}
    for name in _tied_names:
        name = name.split(".")
        name, param_name = ".".join(name[:-1]), name[-1]
        mod = model.get_submodule(name)
        param = getattr(mod, param_name)

        _tied_params[id(param)] = None  # placeholder for the param first

    # build param_init_fn for the case with tied params
    def param_init_fn_tied_param(module: torch.nn.Module):
        # track which params to tie
        # - usually only 1, but for completeness consider > 1
        params_to_tie = defaultdict(list)
        for n, param in module.named_parameters(recurse=False):
            if id(param) in _tied_params:
                params_to_tie[id(param)].append(n)

        # call the param init fn, which potentially re-allocates the
        # parameters
        module = param_init_fn(module)

        # search the parameters again and tie them up again
        for id_key, _param_names in params_to_tie.items():
            for param_name in _param_names:
                param = _tied_params[id_key]
                if param is None:
                    # everything will be tied to the first time the
                    # param is observed
                    _tied_params[id_key] = getattr(module, param_name).clone()
                else:
                    setattr(module, param_name, nn.Parameter(param))  # tie

        return module

    return param_init_fn_tied_param


@torch.no_grad()
def offload_model(model: torch.nn.Module):
    assert isinstance(model, FSDP)
    _lazy_init(model, model)
    assert model._is_root, "Only support root model offloading to CPU"
    for handle in model._all_handles:
        if handle._offload_params:
            continue
        flat_param = handle.flat_param
        assert (
            flat_param.data.data_ptr() == flat_param._local_shard.data_ptr()
            and id(flat_param.data) != id(flat_param._local_shard)
            and flat_param.data.size() == flat_param._local_shard.size()
        )
        handle.flat_param_to(torch.device("cpu"), non_blocking=True)
        flat_param._local_shard = flat_param.data
    torch.cuda.empty_cache()


@torch.no_grad()
def reload_model(model: FSDP):
    assert isinstance(model, FSDP)
    _lazy_init(model, model)
    assert model._is_root, "Only support root model loading to GPU"
    device_id = torch.cuda.current_device()
    for handle in model._all_handles:
        if handle._offload_params:
            continue
        flat_param = handle.flat_param
        handle.flat_param_to(torch.device(f"cuda:{device_id}"), non_blocking=True)
        flat_param._local_shard = flat_param.data


@torch.no_grad()
def offload_optimizer(optimizer: torch.optim.Optimizer):
    if not optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu", non_blocking=True)


def reload_optimizer(optimizer: torch.optim.Optimizer):
    if not optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cuda", non_blocking=True)


def iter_fsdp1_state_dict(model: FSDP | None = None, sharded_state_dict: dict | None = None):
    """Iterate over the state dict of an FSDP model.

    This function will gather the weights key by key, avoiding materializing
    the entire state dict at once and thus is memory efficient.
    """
    # NOTE (zhanghan): We used ShardedTensor.gather and dist.broadcast, where
    #                  we assume the model's sharding process group is built
    #                  with dist.init_process_group and its replicate group
    #                  is not. This is the default implementation in Rlite
    #                  parallel, since Rlite_init_process_group will lead to
    #                  hanging when initializing the FSDP instance.
    if model is None and sharded_state_dict is None:
        raise ValueError("Either `model` or `sharded_state_dict` must be provided.")

    if sharded_state_dict is None:
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            sharded_state_dict = model.state_dict()

    for k, v in sharded_state_dict.items():
        gathered_value = torch.empty(
            v.shape,
            device=torch.device("cuda"),
            dtype=v.dtype,
        )
        v.gather(out=gathered_value if dist.get_rank() == 0 else None)
        dist.broadcast(gathered_value, src=0)
        yield k, gathered_value


def param_init_preserve_buffers(module: torch.nn.Module):
    """Initialize the parameters of the module, but preserve the buffers."""
    buffers = {}
    for name, param in module.named_buffers():
        buffers[name] = param
    module = module.to_empty(device=torch.device("cuda"), recurse=False)
    for name, param in module.named_buffers():
        param.data.copy_(buffers[name].data)
    return module
