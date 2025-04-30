from torch.utils.tensorboard import SummaryWriter


class DummySummaryWriter(SummaryWriter):
    def __getattr__(self, name: str):
        if name.startswith("add_"):
            return self.dummy_call
        return super().__getattr__(name)

    def dummy_call(self, *args, **kwargs):
        pass
