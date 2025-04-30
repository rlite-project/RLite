from __future__ import annotations

from typing import TYPE_CHECKING

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

if TYPE_CHECKING:
    from rlite.resman import ResourceManager


class ResourceBundle:
    """A resource bundle that contains the resource request of a consumer."""

    def __init__(self, num_cpus: int, num_gpus: float):
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.__total_num_cpus = num_cpus
        self.__total_num_gpus = num_gpus

        # Resource allocation info
        self.bundle_id_in_placement_group = None
        self.node_id = None
        self.device_id = None
        self.placement_group = None
        self.resource_manager = None

    @property
    def total_num_cpus(self) -> int:
        return self.__total_num_cpus

    @property
    def total_num_gpus(self) -> int:
        return self.__total_num_gpus

    def to_dict(self, include_device_info: bool = False) -> dict:
        if include_device_info and self.is_resource_allocated:
            ans = {
                "node_id": self.node_id,
                "device_id": self.device_id,
                "CPU": self.num_cpus,
                "GPU": self.num_gpus,
            }
        else:
            ans = {"CPU": self.num_cpus, "GPU": self.num_gpus}

        return {k: v for k, v in ans.items() if v is not None}

    def bind_resources(
        self,
        *,
        device_id: int,
        bundle_id: int,
        node_id: str,
        placement_group: PlacementGroup,
        resource_manager: ResourceManager,
    ):
        if not isinstance(node_id, str) or len(node_id) == 0:
            raise ValueError("Invalid node ID!", node_id)
        if not isinstance(device_id, int) or device_id < 0:
            raise ValueError("Invalid device ID!", device_id)
        if not isinstance(bundle_id, int) or bundle_id < 0:
            raise ValueError("Invalid bundle ID!", bundle_id)
        if not isinstance(placement_group, PlacementGroup):
            raise ValueError("Invalid placement group!", placement_group)

        self.device_id = device_id
        self.bundle_id_in_placement_group = bundle_id
        self.node_id = node_id
        self.placement_group = placement_group
        self.resource_manager = resource_manager

    def remote(self, actor: any, **options):
        if not self.is_resource_allocated:
            raise RuntimeError("This resource bundle's resources are not allocated!")
        options.update(
            {
                "scheduling_strategy": PlacementGroupSchedulingStrategy(
                    placement_group=self.placement_group,
                    placement_group_bundle_index=self.bundle_id_in_placement_group,
                ),
                "num_cpus": self.num_cpus,
                "num_gpus": self.num_gpus,
            }
        )
        return ray.remote(actor).options(**options)

    @property
    def is_resource_allocated(self) -> bool:
        return self.placement_group is not None

    def __add__(self, other: "ResourceBundle") -> "ResourceBundle":
        if self.is_resource_allocated or other.is_resource_allocated:
            if self.node_id != other.node_id or self.device_id != other.device_id:
                raise RuntimeError("Cannot add resource bundles from different nodes or devices!")

        new_bundle = ResourceBundle(
            num_cpus=self.num_cpus + other.num_cpus,
            num_gpus=self.num_gpus + other.num_gpus,
        )
        if self.is_resource_allocated:
            new_bundle.bind_resources(
                bundle_id=self.bundle_id_in_placement_group,
                node_id=self.node_id,
                device_id=self.device_id,
                placement_group=self.placement_group,
                resource_manager=self.resource_manager,
            )
        return new_bundle

    def __sub__(self, other: "ResourceBundle") -> "ResourceBundle":
        if self.is_resource_allocated and other.is_resource_allocated:
            if self.node_id != other.node_id or self.device_id != other.device_id:
                raise RuntimeError(
                    "Cannot subtract resource bundles from different nodes or devices!"
                )

        new_bundle = ResourceBundle(
            num_cpus=self.num_cpus - other.num_cpus,
            num_gpus=self.num_gpus - other.num_gpus,
        )
        if self.is_resource_allocated:
            new_bundle.bind_resources(
                bundle_id=self.bundle_id_in_placement_group,
                node_id=self.node_id,
                device_id=self.device_id,
                placement_group=self.placement_group,
                resource_manager=self.resource_manager,
            )
        return new_bundle

    def __iadd__(self, other: "ResourceBundle") -> "ResourceBundle":
        if self.is_resource_allocated or other.is_resource_allocated:
            if self.node_id != other.node_id or self.device_id != other.device_id:
                raise RuntimeError("Cannot add resource bundles from different nodes or devices!")

        self.num_cpus += other.num_cpus
        self.num_gpus += other.num_gpus
        return self

    def __isub__(self, other: "ResourceBundle") -> "ResourceBundle":
        if self.is_resource_allocated and other.is_resource_allocated:
            if self.node_id != other.node_id or self.device_id != other.device_id:
                raise RuntimeError(
                    "Cannot subtract resource bundles from different nodes or devices!"
                )

        self.num_cpus -= other.num_cpus
        self.num_gpus -= other.num_gpus
        return self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResourceBundle):
            raise NotImplementedError()
        return self.num_cpus == other.num_cpus and self.num_gpus == other.num_gpus

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, ResourceBundle):
            raise NotImplementedError()
        return self.num_cpus > other.num_cpus and self.num_gpus > other.num_gpus

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, ResourceBundle):
            raise NotImplementedError()
        return self.num_cpus >= other.num_cpus and self.num_gpus >= other.num_gpus

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ResourceBundle):
            raise NotImplementedError()
        return not self >= other

    def __le__(self, other: object) -> bool:
        if not isinstance(other, ResourceBundle):
            raise NotImplementedError()
        return not self > other

    def __repr__(self) -> str:
        fields = ["node_id", "device_id", "num_cpus", "num_gpus"]
        formatted_fields = ", ".join(f"{field}={getattr(self, field)}" for field in fields)
        return f"ResourceBundle({formatted_fields})"
