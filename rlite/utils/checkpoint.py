from __future__ import annotations

import json
from pathlib import Path
from typing import Generator

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


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
