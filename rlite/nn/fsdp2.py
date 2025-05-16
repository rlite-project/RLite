import abc

import torch

from rlite.nn.train_module import BaseHuggingFaceTrainModule, BaseTrainModule


class Fsdp2TrainModule(BaseTrainModule, abc.ABC):
    @property
    @abc.abstractmethod
    def non_split_modules(self) -> set[type[torch.nn.Module]]:
        """Get the modules that should not be split across FSDP instances."""
        raise NotImplementedError()

    def postprocess_output_named_parameters(
        self,
        name: str,
        param: torch.Tensor,
        full_state_dict: dict[str, torch.Tensor]
    ) -> tuple[str, torch.Tensor]:
        return name, param.full_tensor()


class HuggingFaceFsdp2TrainModule(BaseHuggingFaceTrainModule, Fsdp2TrainModule, abc.ABC):
    """Rlite FSDP Module for Hugging Face models."""
