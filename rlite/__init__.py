import torch

import rlite.third_party.transformers.monkey_patch  # noqa
from rlite.inference import RliteInferenceEngine
from rlite.resman import ResourceManager
from rlite.train import RliteTrainEngine
from rlite.utils import NeedParallel


def init(
    network_interface: str = "eth0",
    dedup_logs: bool = False,
    debug_mode: str = "debugpy",
    debug_post_mortem: bool = False,
):
    import os
    import socket

    import ray

    assert not ray.is_initialized(), "Let rlite initialize the ray!"

    assert debug_mode in ["debugpy", "pdb"], f"Invalid debug mode: {debug_mode}"

    ray_env = {"env_vars": {"RAY_DEDUP_LOGS": "0"}}

    if debug_mode == "pdb":
        os.environ["RAY_DEBUG"] = "legacy"
        ray_env["env_vars"]["RAY_DEBUG"] = "legacy"
        ray_env["env_vars"]["REMOTE_PDB_HOST"] = socket.gethostbyname(socket.gethostname())

    if debug_post_mortem:
        os.environ["RAY_DEBUG_POST_MORTEM"] = "1"
        ray_env["env_vars"]["RAY_DEBUG_POST_MORTEM"] = "1"

    if dedup_logs:
        ray_env["env_vars"]["RAY_DEDUP_LOGS"] = "1"

    ray.init(runtime_env=ray_env)

    import rlite.resman

    rlite.resman.RESMAN = ResourceManager(network_interface=network_interface)


def device(device_tag: str | torch.device):
    import os

    from accelerate import init_empty_weights
    from loguru import logger

    if isinstance(device_tag, str):
        device_tag = device_tag.lower()

    if device_tag in ["meta", torch.device("meta")]:
        if not os.environ.get("RLITE_META_DEVICE_WARNING", "1") in ["1", "true"]:
            logger.warning(
                "You are using meta device. This will call `accelerate.init_empty_weights()` "
                "which will leave all buffers and tensors in CPU. To suppress this warning, "
                "set environment variable `RLITE_META_DEVICE_WARNING=0`."
            )
        return init_empty_weights()
    return torch.device(device_tag)


__all__ = [
    "NeedParallel",
    "RliteTrainEngine",
    "RliteInferenceEngine",
    "ResourceManager",
    "device"
]
