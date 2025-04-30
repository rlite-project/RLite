import abc
from functools import partial

from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from rlite.nn.train_module import BaseHuggingFaceTrainModule, BaseTrainModule


class FsdpTrainModule(BaseTrainModule, abc.ABC):
    @abc.abstractmethod
    def get_fsdp_wrap_policy(self):
        """Get the FSDP wrap policy.

        This method is called by the `FsdpRayTrainDataParallel` class to get
        the FSDP auto wrap policy.
        """
        raise NotImplementedError()


class HuggingFaceFsdpTrainModule(BaseHuggingFaceTrainModule, FsdpTrainModule):
    """Rlite FSDP Module for Hugging Face models."""

    def get_fsdp_wrap_policy(self):
        if len(self.non_split_modules) == 0:
            raise NotImplementedError()
        return partial(transformer_auto_wrap_policy, transformer_layer_cls=self.non_split_modules)
