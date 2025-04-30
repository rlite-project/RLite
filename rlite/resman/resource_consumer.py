from __future__ import annotations

from typing import TYPE_CHECKING

from rlite.resman.resource_bundle import ResourceBundle

if TYPE_CHECKING:
    from rlite.resman.resource_manager import ResourceManager


class ResourceConsumer:
    """Interface for any class that consumes resources."""

    def __init__(self, resource_manager: ResourceManager | None = None):
        self._consumer_id = None
        self._resource_manager = None
        self._resource_bundles: list[ResourceBundle] = None

        if resource_manager is not None:
            resource_manager.register(self)

    def is_registered(self, manager: ResourceManager | None = None) -> bool:
        if not bool(self._consumer_id):  # not registered if "" or None
            return False
        if manager is not None and self._resource_manager is not manager:
            return False
        return True

    def is_resource_allocated(self) -> bool:
        return bool(self._resource_bundles)  # not allocated if [] or None

    def is_colocate_with(self, other: ResourceConsumer) -> bool:
        if self._resource_manager is not other._resource_manager:
            return False
        if len(self._resource_bundles) != len(other._resource_bundles):
            return False
        for bundle1, bundle2 in zip(self._resource_bundles, other._resource_bundles):
            if not bundle1.bundle_id_in_placement_group == bundle2.bundle_id_in_placement_group:
                return False
        return True
