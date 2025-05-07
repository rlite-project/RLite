from .resource_bundle import ResourceBundle
from .resource_consumer import ResourceConsumer
from .resource_manager import RESMAN, ResourceManager


def available_gpus() -> list[int]:
    return RESMAN.available_resource_bundles(resource_type="gpu")


def total_gpus() -> int:
    return len(RESMAN.resource_bundles)


__all__ = [
    "ResourceBundle",
    "ResourceConsumer",
    "ResourceManager",
    "available_gpus",
    "total_gpus",
]
