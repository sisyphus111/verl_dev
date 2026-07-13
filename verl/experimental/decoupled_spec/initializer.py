from __future__ import annotations

import asyncio
import copy
from typing import Any

import ray
from omegaconf import DictConfig, OmegaConf

from verl.experimental.decoupled_spec.config import (
    DraftConfig,
    build_draft_model_config,
    build_draft_rollout_config,
    get_draft_config,
    validate_decoupled_spec_config,
)
from verl.experimental.decoupled_spec.topology import (
    create_decoupled_spec_topology,
    normalize_decoupled_spec_endpoint_infos,
    sorted_decoupled_spec_endpoints,
)
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import split_resource_pool
from verl.utils.device import is_torch_npu_available
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangReplica


def _create_verifier_replicas(
    *,
    rollout_replica_class: type,
    rollout_config: RolloutConfig,
    model_config: HFModelConfig,
    num_replicas: int,
) -> list[Any]:
    return [
        rollout_replica_class(
            replica_rank=replica_rank,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=rollout_config.n_gpus_per_node,
        )
        for replica_rank in range(num_replicas)
    ]


def _create_drafter_replicas(
    *,
    rollout_config: RolloutConfig,
    model_config: HFModelConfig,
    draft_config: DraftConfig,
) -> tuple[list[SGLangReplica], list[RayResourcePool]]:
    _check_draft_resources_available(draft_config)
    draft_rollout_config = build_draft_rollout_config(rollout_config, draft_config)
    draft_model_config = build_draft_model_config(model_config, draft_config)
    draft_resource_pool = RayResourcePool(
        process_on_nodes=[draft_config.ngpus_per_node] * draft_config.nnodes,
        use_gpu=True,
        max_colocate_count=3,
        name_prefix="decoupled_spec_draft_pool",
    )
    split_resource_pools = split_resource_pool(draft_resource_pool, draft_config.tp_size)
    if len(split_resource_pools) != draft_config.num_drafters:
        raise RuntimeError(
            f"Expected {draft_config.num_drafters} drafter resource pools, got {len(split_resource_pools)}"
        )

    draft_replicas = [
        SGLangReplica(
            replica_rank=replica_rank,
            config=draft_rollout_config,
            model_config=draft_model_config,
            gpus_per_node=draft_config.ngpus_per_node,
        )
        for replica_rank in range(draft_config.num_drafters)
    ]
    return draft_replicas, split_resource_pools


def _check_draft_resources_available(draft_config: DraftConfig):
    available_resources = ray._private.state.available_resources_per_node()
    alive_node_ids = {node["NodeID"] for node in ray.nodes() if node.get("Alive")}
    eligible_nodes = []
    for node_id in alive_node_ids:
        node_resources = available_resources.get(node_id, {})
        available_gpus = int(node_resources.get("GPU", node_resources.get("NPU", 0)))
        if available_gpus >= draft_config.ngpus_per_node:
            eligible_nodes.append(node_id)

    if len(eligible_nodes) < draft_config.nnodes:
        raise ValueError(
            f"Need {draft_config.nnodes} free GPU nodes with at least "
            f"{draft_config.ngpus_per_node} GPUs each for decoupled-spec drafters, "
            f"but found {len(eligible_nodes)}"
        )


async def _setup_verifier_replicas(
    *,
    verifier_replicas: list[Any],
    worker_group: RayWorkerGroup | None,
    rollout_resource_pool: RayResourcePool | None,
):
    if worker_group:
        for replica in verifier_replicas:
            replica.rollout_mode = RolloutMode.HYBRID
            replica.workers = worker_group.workers[
                replica.world_size * replica.replica_rank : replica.world_size * (replica.replica_rank + 1)
            ]
        return

    verifier_resource_pools = None
    if rollout_resource_pool is not None:
        split_sizes = [replica.world_size for replica in verifier_replicas]
        verifier_resource_pools = split_resource_pool(rollout_resource_pool, split_sizes)
        if len(verifier_resource_pools) != len(verifier_replicas):
            raise ValueError(
                f"Expected {len(verifier_replicas)} verifier resource pools, "
                f"got {len(verifier_resource_pools)}"
            )

    for index, replica in enumerate(verifier_replicas):
        replica.rollout_mode = RolloutMode.STANDALONE
        if verifier_resource_pools is not None:
            replica.resource_pool = verifier_resource_pools[index]
        else:
            resource_pool_name = (
                f"rollout_pool_{replica.replica_rank}"
                if not replica.is_reward_model
                else f"rollout_pool_reward_{replica.replica_rank}"
            )
            resource_pool_spec = {
                resource_pool_name: [replica.gpus_per_replica_node] * replica.nnodes,
            }
            resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=None)
            resource_pool_manager.create_resource_pool()
            replica.resource_pool = resource_pool_manager.resource_pool_dict[resource_pool_name]

        use_gpu = replica.rollout_worker_use_gpu()
        worker_group = RayWorkerGroup(
            resource_pool=replica.resource_pool,
            ray_cls_with_init=replica.get_ray_class_with_init_args(),
            bin_pack=False,
            name_prefix=f"rollout_standalone_{replica.replica_rank}"
            if not replica.is_reward_model
            else f"rollout_reward_standalone_{replica.replica_rank}",
            use_gpu=use_gpu,
            device_name="cuda" if not is_torch_npu_available(check_device=False) else "npu",
        )
        replica.workers = worker_group.workers


async def initialize_decoupled_spec_rollout_servers(
    *,
    config: DictConfig,
    rollout_config: RolloutConfig,
    model_config: HFModelConfig,
    rollout_replica_class: type,
    num_replicas: int,
    worker_group: RayWorkerGroup | None,
    rollout_resource_pool: RayResourcePool | None,
) -> tuple[list[Any], list[SGLangReplica]]:
    draft_config = get_draft_config(config)
    verifier_rollout_config = (
        OmegaConf.create(OmegaConf.to_container(rollout_config, resolve=False))
        if isinstance(rollout_config, DictConfig)
        else OmegaConf.create(copy.deepcopy(rollout_config))
    )
    validate_decoupled_spec_config(verifier_rollout_config, draft_config)

    verifier_replicas = _create_verifier_replicas(
        rollout_replica_class=rollout_replica_class,
        rollout_config=verifier_rollout_config,
        model_config=model_config,
        num_replicas=num_replicas,
    )
    await _setup_verifier_replicas(
        verifier_replicas=verifier_replicas,
        worker_group=worker_group,
        rollout_resource_pool=rollout_resource_pool,
    )
    draft_replicas, draft_resource_pools = _create_drafter_replicas(
        rollout_config=rollout_config,
        model_config=model_config,
        draft_config=draft_config,
    )
    for replica, resource_pool in zip(draft_replicas, draft_resource_pools, strict=True):
        replica.rollout_mode = RolloutMode.STANDALONE
        replica.resource_pool = resource_pool

        use_gpu = replica.rollout_worker_use_gpu()
        worker_group = RayWorkerGroup(
            resource_pool=replica.resource_pool,
            ray_cls_with_init=replica.get_ray_class_with_init_args(),
            bin_pack=False,
            name_prefix=f"decoupled_draft_rollout_standalone_{replica.replica_rank}",
            use_gpu=use_gpu,
            device_name="cuda" if not is_torch_npu_available(check_device=False) else "npu",
        )
        replica.workers = worker_group.workers

    await asyncio.gather(
        asyncio.gather(*[replica.prepare_server_actors() for replica in verifier_replicas]),
        asyncio.gather(
            *[replica.prepare_server_actors(server_name_prefix="sglang_draft_server") for replica in draft_replicas]
        ),
    )
    verifier_dp_size = int(verifier_rollout_config.data_parallel_size)
    topology = create_decoupled_spec_topology(
        num_verifier_replicas=len(verifier_replicas),
        verifier_endpoints_per_replica=verifier_dp_size,
        num_draft_replicas=len(draft_replicas),
    )

    verifier_server_configs = [
        endpoint_config.to_server_config(
            algorithm="DECOUPLED_VERIFY",
            speculative_num_steps=draft_config.speculative_num_steps,
            trace_dir=draft_config.trace_dir,
        )
        for endpoint_config in topology.verifier_configs
    ]
    drafter_server_configs = [
        endpoint_config.to_server_config(
            algorithm="DECOUPLED_DRAFT",
            speculative_num_steps=draft_config.speculative_num_steps,
            trace_dir=draft_config.trace_dir,
        )
        for endpoint_config in topology.drafter_configs
    ]
    for replica, server_config in zip(verifier_replicas, verifier_server_configs, strict=True):
        replica.set_decoupled_spec_config(server_config)
    for replica, server_config in zip(draft_replicas, drafter_server_configs, strict=True):
        replica.set_decoupled_spec_config(server_config)

    await asyncio.gather(
        *[replica.launch_prepared_servers() for replica in draft_replicas],
        *[replica.launch_prepared_servers() for replica in verifier_replicas],
    )

    verifier_endpoint_infos_by_replica, drafter_endpoint_infos_by_replica = await asyncio.gather(
        asyncio.gather(*[replica.get_decoupled_spec_endpoint_infos() for replica in verifier_replicas]),
        asyncio.gather(*[replica.get_decoupled_spec_endpoint_infos() for replica in draft_replicas]),
    )
    verifier_endpoint_infos = normalize_decoupled_spec_endpoint_infos(
        [info for infos in verifier_endpoint_infos_by_replica for info in infos]
    )
    drafter_endpoint_infos = normalize_decoupled_spec_endpoint_infos(
        [info for infos in drafter_endpoint_infos_by_replica for info in infos]
    )

    verifier_result_endpoints = sorted_decoupled_spec_endpoints(
        verifier_endpoint_infos,
        role="verifier",
        expected_count=sum(config.endpoint_count for config in topology.verifier_configs),
    )
    drafter_control_endpoints = sorted_decoupled_spec_endpoints(
        drafter_endpoint_infos,
        role="drafter",
        expected_count=sum(config.endpoint_count for config in topology.drafter_configs),
    )

    verifier_configure_results, drafter_configure_results = await asyncio.gather(
        asyncio.gather(
            *[replica.configure_decoupled_spec_peers(drafter_control_endpoints) for replica in verifier_replicas]
        ),
        asyncio.gather(
            *[replica.configure_decoupled_spec_peers(verifier_result_endpoints) for replica in draft_replicas]
        ),
    )
    failures = [
        message
        for success, message in [*verifier_configure_results, *drafter_configure_results]
        if not success
    ]
    if failures:
        raise RuntimeError("failed to configure decoupled-spec peers: " + " | ".join(failures))

    return verifier_replicas, draft_replicas
