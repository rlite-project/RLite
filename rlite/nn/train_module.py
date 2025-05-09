from __future__ import annotations

import abc
import os
from functools import cached_property
from pathlib import Path

import torch

from rlite.utils.distributed import SerializableModule


class BaseTrainModule(SerializableModule, abc.ABC):
    def init_hook(self):
        """Hook that is called after the model is initialized."""

    def materialization_source(self) -> dict[str, torch.Tensor] | dict[str, callable] | str:
        """Materialization source.

        Returns:
            source: The source of the weights. Can be a state dict (unsharded
                    weights on CPU), a dict of `param_init_fn` (iteratively
                    initialize the weights), or a path to the checkpoint files.
        """
        raise NotImplementedError()

    def on_weights_materialized(self):
        """Hook that is called after the model's weights are materialized.

        This method is designed for logic that happens after the model's
        weights are materialized on the GPU.
        """


class BaseHuggingFaceTrainModule(BaseTrainModule):
    def init_hook(self):
        """Infer the materialization source."""

        def get_latest_directory(directory: Path) -> Path | None:
            if not directory.exists():
                return None
            directories = [d for d in directory.iterdir() if d.is_dir()]
            if not directories:
                return None
            return max(directories, key=lambda d: d.stat().st_mtime)

        if Path(self.config.name_or_path).exists():
            self._materialization_source = self.config.name_or_path
            return

        # Infer the default path for cache
        default_cache_root = Path.home() / ".cache/huggingface/hub"
        cache_root = Path(os.getenv("HF_HOME", os.getenv("HF_HUB_CACHE", default_cache_root)))
        cache_name = self.config.name_or_path.replace("/", "--")
        cache_path = get_latest_directory(cache_root / f"models--{cache_name}" / "snapshots")

        # Download if no safetensors found (TODO: support other formats)
        if len(list(cache_path.glob("*.safetensors"))) == 0:  # No safetensors found, download
            import huggingface_hub
            huggingface_hub.snapshot_download(self.config.name_or_path)
            cache_path = get_latest_directory(cache_root / f"models--{cache_name}" / "snapshots")
            if cache_path is None:
                raise ValueError(f"Failed to download the model from {self.config.name_or_path}")

        self._materialization_source = str(cache_path)

    def materialization_source(self) -> str:
        """Always return an existing path in the file system."""
        return self._materialization_source

    @cached_property
    def non_split_modules(self) -> set[type[torch.nn.Module]]:
        if hasattr(self, "_no_split_modules"):
            class_names = self._no_split_modules
            no_split_modules = set()
            for module in self.modules():
                if module.__class__.__name__ in class_names:
                    no_split_modules.add(module.__class__)
            return no_split_modules

        elif hasattr(self, "get_decoder"):
            return {self.get_decoder().layers[0].__class__}

        elif hasattr(self, "layers"):
            return {self.layers[0].__class__}

        return set()
