import abc

import torch

from rlite.nn.train_module import BaseHuggingFaceTrainModule, BaseTrainModule


class Fsdp2TrainModule(BaseTrainModule, abc.ABC):
    @property
    @abc.abstractmethod
    def non_split_modules(self) -> set[type[torch.nn.Module]]:
        """Get the modules that should not be split across FSDP instances."""
        raise NotImplementedError()


class HuggingFaceFsdp2TrainModule(BaseHuggingFaceTrainModule, Fsdp2TrainModule, abc.ABC):
    """Oone FSDP Module for Hugging Face models."""
