from __future__ import annotations

import abc
from functools import cached_property

import torch

from rlite.utils.distributed import SerializableModule


class BaseTrainModule(SerializableModule, abc.ABC):
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
    def materialization_source(self) -> str:
        return self.config.name_or_path

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
