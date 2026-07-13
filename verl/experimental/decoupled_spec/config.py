from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

from verl.base_config import BaseConfig


@dataclass
class DraftConfig(BaseConfig):
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    load_format: str = "auto"
    quantization: Optional[str] = "bfloat16"
    nnodes: int = 1
    ngpus: int = 0
    tp_size: int = 1
    speculative_num_steps: int = 3
    trace_dir: Optional[str] = None
    output_dir: Optional[str] = None

    def __post_init__(self):
        if not self.model_path:
            raise ValueError("draft.model_path must be set when rollout.enable_decoupled_spec=True")
        if self.nnodes <= 0:
            raise ValueError("draft.nnodes must be > 0")
        if self.ngpus <= 0:
            raise ValueError("draft.ngpus must be > 0")
        if self.tp_size <= 0:
            raise ValueError("draft.tp_size must be > 0")
        if self.speculative_num_steps <= 0:
            raise ValueError("draft.speculative_num_steps must be > 0")
        if self.ngpus % self.tp_size != 0:
            raise ValueError("draft.ngpus must be divisible by draft.tp_size")
        if self.ngpus % self.nnodes != 0:
            raise ValueError("draft.ngpus must be divisible by draft.nnodes")
        if (self.ngpus // self.nnodes) % self.tp_size != 0:
            raise ValueError("draft.ngpus / draft.nnodes must be divisible by draft.tp_size")
        _draft_quantization_to_rollout(self.quantization)

    @property
    def num_drafters(self) -> int:
        if self.tp_size <= 0:
            return 0
        return self.ngpus // self.tp_size

    @property
    def ngpus_per_node(self) -> int:
        if self.nnodes <= 0:
            return 0
        return self.ngpus // self.nnodes


def get_draft_config(config: DictConfig | dict[str, Any] | None) -> DraftConfig:
    if config is None:
        return DraftConfig()

    draft_cfg = config.get("draft") if isinstance(config, (DictConfig, dict)) else None
    if draft_cfg is None:
        return DraftConfig()

    structured = OmegaConf.structured(DraftConfig)
    merged = OmegaConf.merge(structured, draft_cfg)
    return OmegaConf.to_object(merged)


def _copy_config(config: DictConfig | dict[str, Any]) -> DictConfig:
    if isinstance(config, DictConfig):
        return OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    return OmegaConf.create(copy.deepcopy(config))


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _to_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _sglang_engine_kwargs(rollout_config: Any) -> Any:
    engine_kwargs = _config_get(rollout_config, "engine_kwargs", {}) or {}
    return _config_get(engine_kwargs, "sglang", {}) or {}


def _draft_quantization_to_rollout(quantization: Optional[str]) -> tuple[str, Optional[str]]:
    if quantization is None:
        return "bfloat16", None

    normalized = str(quantization).lower()
    if normalized in {"none", "null", "bf16", "bfloat16"}:
        return "bfloat16", None
    if normalized in {"fp16", "float16"}:
        return "float16", None
    if normalized == "fp8":
        return "bfloat16", "fp8"

    raise ValueError(
        "draft.quantization must be one of: bfloat16, bf16, float16, fp16, fp8, null; "
        f"got {quantization!r}"
    )


def build_draft_model_config(
    model_config: DictConfig | dict[str, Any],
    draft_config: DraftConfig,
) -> DictConfig:
    cfg = _copy_config(model_config)
    cfg.path = draft_config.model_path
    cfg.local_path = None
    cfg.hf_config_path = draft_config.model_path
    cfg.local_hf_config_path = None
    cfg.tokenizer_path = draft_config.tokenizer_path or draft_config.model_path
    cfg.local_tokenizer_path = None
    cfg.hf_config = None
    cfg.generation_config = None
    cfg.tokenizer = None
    cfg.processor = None
    cfg.architectures = None
    return cfg


def build_draft_rollout_config(
    rollout_config: DictConfig | dict[str, Any],
    draft_config: DraftConfig,
) -> DictConfig:
    cfg = _copy_config(rollout_config)
    cfg.name = "sglang"
    cfg.nnodes = draft_config.nnodes
    cfg.n_gpus_per_node = draft_config.ngpus_per_node
    cfg.tensor_model_parallel_size = draft_config.tp_size
    cfg.data_parallel_size = 1
    cfg.pipeline_model_parallel_size = 1
    cfg.expert_parallel_size = 1
    cfg.load_format = draft_config.load_format
    cfg.dtype, cfg.quantization = _draft_quantization_to_rollout(draft_config.quantization)
    cfg.quantization_config_file = None
    cfg.enable_decoupled_spec = True
    return cfg


def validate_decoupled_spec_config(rollout_config: Any, draft_config: DraftConfig):
    if rollout_config.name != "sglang":
        raise ValueError("rollout.enable_decoupled_spec=True only supports rollout.name=sglang")
    if rollout_config.pipeline_model_parallel_size != 1:
        raise ValueError("decoupled speculative decoding currently requires verifier pipeline_model_parallel_size == 1")
    if int(rollout_config.data_parallel_size) > 1:
        sglang_kwargs = _sglang_engine_kwargs(rollout_config)
        if not _to_bool(_config_get(sglang_kwargs, "enable_dp_attention", False)):
            raise ValueError(
                "decoupled verifier data_parallel_size > 1 requires "
                "actor_rollout_ref.rollout.engine_kwargs.sglang.enable_dp_attention=True"
            )
    mtp_config = rollout_config.get("mtp", None)
    if mtp_config is not None and mtp_config.get("enable", False) and mtp_config.get("enable_rollout", False):
        raise ValueError("decoupled speculative decoding cannot be enabled together with MTP rollout")
    if draft_config.tp_size > draft_config.ngpus_per_node:
        raise ValueError(
            f"draft.tp_size ({draft_config.tp_size}) must be <= draft.ngpus / draft.nnodes "
            f"({draft_config.ngpus_per_node})"
        )
