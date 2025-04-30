import abc

from rlite.utils.distributed import SerializableModule


class BaseInferenceModule(SerializableModule, abc.ABC):
    @abc.abstractmethod
    def get_checkpoint_path(self) -> str:
        """Get the checkpoint path."""
        raise NotImplementedError()
