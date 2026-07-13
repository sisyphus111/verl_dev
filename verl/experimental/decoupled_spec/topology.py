from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

_MISSING = object()


@dataclass
class DecoupledSpecLaunchConfig:
    rank_base: int
    endpoint_count: int

    def to_server_config(
        self,
        *,
        algorithm: str,
        speculative_num_steps: int,
        trace_dir: Optional[str] = None,
    ) -> dict:
        return {
            "algorithm": algorithm,
            "rank_base": self.rank_base,
            "speculative_num_steps": speculative_num_steps,
            "spec_trace_dir": trace_dir,
        }


@dataclass
class DecoupledSpecEndpointInfo:
    role: str
    rank: int
    local_dp_rank: int
    bind_endpoint: str


@dataclass
class DecoupledSpecTopology:
    verifier_configs: list[DecoupledSpecLaunchConfig]
    drafter_configs: list[DecoupledSpecLaunchConfig]


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def create_decoupled_spec_topology(
    *,
    num_verifier_replicas: int,
    verifier_endpoints_per_replica: int,
    num_draft_replicas: int,
) -> DecoupledSpecTopology:
    _require_positive("num_verifier_replicas", int(num_verifier_replicas))
    _require_positive(
        "verifier_endpoints_per_replica", int(verifier_endpoints_per_replica)
    )
    _require_positive("num_draft_replicas", int(num_draft_replicas))

    verifier_configs = []
    verifier_rank_base = 0
    for _ in range(num_verifier_replicas):
        verifier_configs.append(
            DecoupledSpecLaunchConfig(
                rank_base=verifier_rank_base,
                endpoint_count=verifier_endpoints_per_replica,
            )
        )
        verifier_rank_base += verifier_endpoints_per_replica

    drafter_configs = [
        DecoupledSpecLaunchConfig(
            rank_base=rank,
            endpoint_count=1,
        )
        for rank in range(num_draft_replicas)
    ]
    return DecoupledSpecTopology(verifier_configs=verifier_configs, drafter_configs=drafter_configs)


def normalize_decoupled_spec_endpoint_infos(raw_infos: list[Any]) -> list[DecoupledSpecEndpointInfo]:
    return [
        DecoupledSpecEndpointInfo(
            role=str(_endpoint_field(info, "role")),
            rank=int(_endpoint_field(info, "rank")),
            local_dp_rank=int(_endpoint_field(info, "local_dp_rank")),
            bind_endpoint=str(_endpoint_field(info, "bind_endpoint")),
        )
        for info in raw_infos
    ]


def sorted_decoupled_spec_endpoints(
    endpoint_infos: list[Any], *, role: str, expected_count: Optional[int] = None
) -> list[str]:
    role_infos = [info for info in endpoint_infos if _endpoint_field(info, "role") == role]
    if expected_count is not None and len(role_infos) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} decoupled-spec {role} endpoints, got {len(role_infos)}"
        )
    if not role_infos:
        raise RuntimeError(f"no decoupled-spec {role} endpoints were published")

    role_infos.sort(key=lambda info: int(_endpoint_field(info, "rank")))
    ranks = [int(_endpoint_field(info, "rank")) for info in role_infos]
    expected_ranks = list(range(len(ranks)))
    if ranks != expected_ranks:
        raise RuntimeError(
            f"decoupled-spec {role} ranks must be zero-based and contiguous: got {ranks}"
        )
    return [str(_endpoint_field(info, "bind_endpoint")) for info in role_infos]


def _endpoint_field(info: Any, field: str, default: Any = _MISSING) -> Any:
    if isinstance(info, dict):
        if field in info:
            return info[field]
    elif hasattr(info, field):
        return getattr(info, field)

    if default is not _MISSING:
        return default
    raise RuntimeError(f"decoupled-spec endpoint info missing field {field!r}: {info!r}")
