import abc

from rlite.interface import BaseWorker


class BaseTrainWorker(BaseWorker, abc.ABC):
    @abc.abstractmethod
    def save(self, checkpoint_path: str):
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self, checkpoint_path: str):
        raise NotImplementedError()

    def getattr(self, name, *args, **kwargs):
        attr = getattr(self, name)
        if callable(attr):
            return attr(*args, **kwargs)
        return attr

    def hasattr(self, name):
        return hasattr(self, name) and callable(getattr(self, name))
