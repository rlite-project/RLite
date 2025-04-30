from __future__ import annotations

import random
import socket
import uuid
from collections import defaultdict
from functools import cached_property
from typing import TYPE_CHECKING, Literal

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from rlite.resman.resource_bundle import ResourceBundle

if TYPE_CHECKING:
    from .resource_consumer import ResourceConsumer


class ResourceManager:
    """Resource manager that manages the resources of the cluster."""

    MAX_COLATE_COUNT = 20

    pg: PlacementGroup = None
    resource_bundles: list[ResourceBundle] = []
    node_resource_info: dict[str, dict[str, float]] = {}
    resource_allocations: dict[str, ResourceBundle] = {}
    allocated_ports: dict[str, list[int]] = {}

    def __init__(
        self,
        *,
        master_addr: str | None = None,
    ):
        if self.is_initialized():
            return

        # Handle if there is no GPU in the cluster
        if "GPU" not in ray.cluster_resources() or ray.cluster_resources()["GPU"] == 0:
            self._gpu_available = False
        else:
            self._gpu_available = True

        # Record the node resource info
        ResourceManager.node_resource_info = {
            node["NodeID"]: {
                "CPU": int(node["Resources"].get("CPU", 0)),
                "GPU": int(node["Resources"].get("GPU", 0)),
                "IP": node["NodeManagerAddress"],
                "NodeID": node["NodeID"],
            }
            for node in ray.nodes()
        }

        # Create the resource bundles
        if self._gpu_available:
            for node in ResourceManager.node_resource_info.values():
                for _ in range(node["GPU"]):
                    ResourceManager.resource_bundles.append(
                        ResourceBundle(num_cpus=node["CPU"] // node["GPU"], num_gpus=1)
                    )
        else:
            for node in ResourceManager.node_resource_info.values():
                ResourceManager.resource_bundles.append(
                    ResourceBundle(num_cpus=node["CPU"], num_gpus=None)
                )

        # Set up the placement group
        self.pg = ray.util.placement_group(
            strategy="PACK",
            bundles=[
                bundle.to_dict(include_device_info=False)
                for bundle in ResourceManager.resource_bundles
            ],
        )
        ray.get(self.pg.ready())

        # Bind actual resources to the resource bundles
        mapping = ray.util.placement_group_table(self.pg)["bundles_to_node_id"]

        device_id_book = defaultdict(int)
        for index, bundle in enumerate(ResourceManager.resource_bundles):
            node_id = mapping[index]
            bundle.bind_resources(
                bundle_id=index,
                device_id=device_id_book[node_id],
                node_id=node_id,
                placement_group=self.pg,
                resource_manager=self,
            )
            device_id_book[node_id] += 1

        self._sort_resource_bundles(master_addr)

    def register(self, consumer: ResourceConsumer, consumer_id: str | None = None):
        if consumer_id is None:
            consumer_id = str(uuid.uuid4())[:16]

        if hasattr(consumer, "_consumer_id") and consumer._consumer_id is not None:
            if consumer._consumer_id != consumer_id:
                raise ValueError("The consumer_id is already registered!")
            return

        # Initialize the consumer
        consumer._resource_manager = self
        consumer._consumer_id = consumer_id
        consumer._resource_bundles = []
        self.resource_allocations[consumer_id] = []

    def unregister(self, consumer: ResourceConsumer):
        for bundle in self.resource_allocations.pop(consumer._consumer_id):
            self.resource_bundles[bundle.bundle_id_in_placement_group] += bundle

        delattr(consumer, "_resource_manager")
        delattr(consumer, "_consumer_id")
        delattr(consumer, "_resource_bundles")

    def allocate_resources(
        self,
        consumer: ResourceConsumer,
        num_gpus: int | None = None,
        colocate_with: ResourceConsumer | None = None,
        not_colocate_with: list[ResourceConsumer] | ResourceConsumer | None = None,
        allow_colocate: bool = False,
    ) -> bool:
        if not self.is_initialized():
            raise ValueError("Resource manager is not initialized!")

        if not_colocate_with is not None:
            if not isinstance(not_colocate_with, list):
                not_colocate_with = [not_colocate_with]
            forbidden_bundle_ids = set([
                bundle.bundle_id_in_placement_group
                for consumer in not_colocate_with
                for bundle in consumer._resource_bundles
            ])
        else:
            forbidden_bundle_ids = set()

        # Colocate with the specified consumer
        if colocate_with is not None:
            # Sanity check
            if not colocate_with.is_registered(manager=self):
                raise ValueError("The specified consumer has not been registered!")
            if not colocate_with.is_resource_allocated():
                raise ValueError(
                    "The specified colocate_with consumer has not been allocated resources!"
                )
            bundles = self._get_bundles_from_colocate_consumer(colocate_with, forbidden_bundle_ids)
        else:
            # TODO: handle requires no gpu
            if num_gpus is None:
                raise ValueError("num_gpus is required when colocate_with is not specified!")

            if num_gpus > self.total_gpus:
                raise ValueError(f"Not enough GPUs available! {num_gpus} > {self.total_gpus}")

            if num_gpus > self.available_resource_bundles("gpu"):
                raise ValueError(
                    f"Not enough GPUs available! "
                    f"{num_gpus} > {self.available_resource_bundles('gpu')}"
                )

            bundles = self._get_bundles_from_gpu_request(
                gpu_request=num_gpus,
                allow_colocate=allow_colocate,
                forbidden_bundle_ids=forbidden_bundle_ids,
            )

        # Assign bundles to the consumer
        for bundle in bundles:
            self.resource_allocations[consumer._consumer_id].append(bundle)
            consumer._resource_bundles.append(bundle)

    def _get_bundles_from_colocate_consumer(
        self,
        colocate_with: ResourceConsumer,
        forbidden_bundle_ids: set[int],
    ) -> list[ResourceBundle]:
        # Ensure the bundles not overlap with the not_colocate_with's bundles
        bundle_ids_to_allocate = set([
            bundle.bundle_id_in_placement_group
            for bundle in colocate_with._resource_bundles
        ])
        assert len(forbidden_bundle_ids & bundle_ids_to_allocate) == 0, (
            "The specified colocate_with consumer has overlapping bundles "
            "with the forbidden bundles!"
        )

        # Use the colocate_with's resource bundles
        bundles = []
        for bundle in colocate_with._resource_bundles:
            new_bundle = ResourceBundle(
                num_cpus=self.colocate_num_cpus,
                num_gpus=self.colocate_num_gpus
            )
            new_bundle.bind_resources(
                bundle_id=bundle.bundle_id_in_placement_group,
                device_id=bundle.device_id,
                node_id=bundle.node_id,
                placement_group=bundle.placement_group,
                resource_manager=self,
            )
            source_bundle = self.get_bundle_by_id(bundle.bundle_id_in_placement_group)
            source_bundle -= new_bundle
            bundles.append(new_bundle)

        return bundles

    def _get_bundles_from_gpu_request(
        self,
        gpu_request: int,
        allow_colocate: bool,
        forbidden_bundle_ids: set[int],
    ) -> list[ResourceBundle]:
        # NOTE: as one bundle can hold at most 1 GPU, we need to allocate
        #       num_gpus bundles
        num_cpus = self.colocate_num_cpus if allow_colocate else 1
        num_gpus = self.colocate_num_gpus if allow_colocate else 1

        # Find the available bundles
        bundles_to_allocate = []
        target_bundle = ResourceBundle(num_cpus=num_cpus, num_gpus=num_gpus)
        for bundle in self.resource_bundles:
            if len(bundles_to_allocate) == gpu_request:
                break
            if bundle.bundle_id_in_placement_group in forbidden_bundle_ids:
                continue
            if bundle >= target_bundle:
                bundles_to_allocate.append(bundle)

        if len(bundles_to_allocate) < gpu_request:
            raise ValueError(
                f"Not enough gpus to allocate! {len(bundles_to_allocate)} < {gpu_request}"
            )

        # Allocate resources
        bundles = []
        for bundle in bundles_to_allocate:
            new_bundle = ResourceBundle(num_cpus=num_cpus, num_gpus=num_gpus)
            new_bundle.bind_resources(
                bundle_id=bundle.bundle_id_in_placement_group,
                device_id=bundle.device_id,
                node_id=bundle.node_id,
                placement_group=bundle.placement_group,
                resource_manager=self,
            )
            bundle -= new_bundle
            bundles.append(new_bundle)

        return bundles

    def deallocate_resources(self, consumer: any, bundles: list[ResourceBundle] | None = None):
        if bundles is not None:
            bundle_ids_to_deallocate = [bundle.bundle_id_in_placement_group for bundle in bundles]
        else:
            bundle_ids_to_deallocate = [
                bundle.bundle_id_in_placement_group
                for bundle in self.resource_allocations[consumer._consumer_id]
            ]

        bundles_to_keep = []
        for bundle in self.resource_allocations[consumer._consumer_id]:
            if bundle.bundle_id_in_placement_group not in bundle_ids_to_deallocate:
                bundles_to_keep.append(bundle)
            else:
                # Reclaim the resources
                self.resource_bundles[bundle.bundle_id_in_placement_group] += bundle
        self.resource_allocations[consumer._consumer_id] = bundles_to_keep
        consumer._resource_bundles = bundles_to_keep

    def available_resource_bundles(
        self, resource_type: Literal["gpu", "cpu"] = "gpu", allow_colocate: bool = True
    ) -> int:
        result = 0
        for bundle in self.resource_bundles:
            if resource_type == "gpu" and bundle.num_gpus is not None:
                threshold = self.colocate_num_gpus if allow_colocate else 1
                if bundle.num_gpus >= threshold:
                    result += 1
            if resource_type == "cpu" and bundle.num_cpus is not None:
                threshold = self.colocate_num_cpus if allow_colocate else 1
                if bundle.num_cpus >= threshold:
                    result += 1
        return result

    def get_free_port(self, bundle: ResourceBundle | int) -> int:
        """Get a free port from the host node of the bundle."""

        if isinstance(bundle, int):
            bundle = self.resource_bundles[bundle]

        def test_port_available(port: int) -> bool:
            def _test_port_available(port: int) -> bool:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    return s.connect_ex(("localhost", port)) != 0

            return ray.get(
                ray.remote(_test_port_available)
                .options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=self.pg,
                        placement_group_bundle_index=bundle.bundle_id_in_placement_group,
                    )
                )
                .remote(port)
            )

        def port_allocated(port: int) -> bool:
            return port in self.allocated_ports.get(bundle.node_id, [])

        if not self.is_initialized():
            raise ValueError("Resource manager is not initialized!")

        port = random.randint(1024, 65535)
        while port_allocated(port) or not test_port_available(port):
            port = random.randint(1024, 65535)

        self.allocated_ports.setdefault(bundle.node_id, []).append(port)

        return port

    def get_ip_addr(self, bundle: ResourceBundle | int) -> str:
        """Get the IP address of the host node of the given bundle."""

        if isinstance(bundle, int):
            bundle = self.resource_bundles[bundle]
        return ResourceManager.node_resource_info[bundle.node_id]["IP"]

    @cached_property
    def colocate_num_gpus(self) -> float:
        return 1 / self.MAX_COLATE_COUNT

    @property
    def colocate_num_cpus(self) -> float:
        return 1.0 / self.MAX_COLATE_COUNT

    @cached_property
    def total_gpus(self) -> float:
        return sum(
            bundle.num_gpus
            for bundle in self.resource_bundles
            if bundle.num_gpus is not None
        )

    def is_initialized(self) -> bool:
        return self.pg is not None

    def get_bundle_by_id(self, bundle_id: int):
        """Fetch the bundle by id, useful after sorting the bundles."""
        for bundle in self.resource_bundles:
            if bundle.bundle_id_in_placement_group == bundle_id:
                return bundle
        raise ValueError(f"Bundle ID {bundle_id} not found!")

    def _sort_resource_bundles(self, master_addr: str | None = None):
        # NOTE (zhanghan): We need to sort the resource bundles by the node id,
        #                  so that the head node will always be the first one,
        #                  and the resource bundles on the same node will be
        #                  grouped together.
        if master_addr is None:
            master_addr = socket.gethostbyname(socket.gethostname())
        for node in self.node_resource_info.values():
            if node["IP"] == master_addr:
                head_node_id = node["NodeID"]
                break
        ResourceManager.resource_bundles = list(
            sorted(
                ResourceManager.resource_bundles,
                key=lambda x: (x.node_id != head_node_id, x.node_id)
            )
        )

    def __del__(self):
        if ray.is_initialized():
            ray.shutdown()


RESMAN: ResourceManager | None = None
