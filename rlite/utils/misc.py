def flatten(item: list[list[any]]) -> list[any]:
    if not hasattr(item, "__iter__") or not hasattr(item, "__getitem__"):
        return item
    return sum([flatten(i) for i in item], [])


def set_random_seed(seed: int):
    import random

    import numpy
    import torch

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
