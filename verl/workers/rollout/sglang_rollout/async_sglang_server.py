# Copyright 2023-2024 SGLang Team
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import dataclasses
import json
import logging
import os
import socket
from typing import Any, Optional

import ray
import sglang
import sglang.srt.entrypoints.engine
import torch
from packaging import version
from ray.actor import ActorHandle
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.entrypoints.http_server import (
    ServerArgs,
    _GlobalState,
    app,
    set_global_state,
)
from sglang.srt.managers.io_struct import (
    ContinueGenerationReqInput,
    GenerateReqInput,
    PauseGenerationReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
)
from sglang.srt.managers.tokenizer_manager import ServerStatus

from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_visible_devices_keyword
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address
from verl.utils.profiler import DistProfiler, build_sglang_profiler_args
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode, RolloutReplica, TokenOutput
from verl.workers.rollout.sglang_rollout.sglang_rollout import _set_envs_and_config
from verl.workers.rollout.utils import get_max_position_embeddings, run_uvicorn

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

visible_devices_keyword = get_visible_devices_keyword()

SGLANG_DP_ATTENTION_AVAILABLE_PORT_COUNT = 6
SGLANG_DP_ATTENTION_ENV_PORT_MAX_COUNT = 30


def _reserve_port_on_address(address: str, port: int) -> socket.socket:
    family = socket.AF_INET6 if is_valid_ipv6_address(address) else socket.AF_INET
    sock = socket.socket(family=family, type=socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if family == socket.AF_INET6 and hasattr(socket, "IPV6_V6ONLY"):
        try:
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        except OSError:
            pass
    try:
        if family == socket.AF_INET6:
            sock.bind((address, port, 0, 0))
        else:
            sock.bind((address, port))
        sock.listen(1)
    except OSError:
        sock.close()
        raise
    return sock


def _get_numbered_env_ports(prefix: str = "PORT", max_count: int = 30) -> list[int]:
    ports = []
    for port_index in range(1, max_count + 1):
        name = f"{prefix}{port_index}"
        raw_port = os.environ.get(name)
        if not raw_port:
            continue
        try:
            port = int(raw_port)
        except ValueError as exc:
            raise ValueError(
                f"{name} must be an integer TCP port, got {raw_port!r}"
            ) from exc
        if port <= 0 or port > 65535:
            raise ValueError(f"{name} is out of TCP port range: {port}")
        ports.append(port)

    if len(set(ports)) != len(ports):
        raise ValueError(
            f"SGLang environment ports from {prefix}1..{prefix}{max_count} "
            f"must be unique: {ports}"
        )
    return ports


def _is_port_available_on_address(address: str, port: int) -> bool:
    try:
        sock = _reserve_port_on_address(address, port)
    except OSError:
        return False
    sock.close()
    return True


def _select_sglang_dp_attention_available_ports(
    address: str,
    replica_rank: int,
    avoid_ports: set[int],
) -> list[int]:
    """Select SGLang DP-attention available_ports from PORT1..PORT30."""
    env_ports = _get_numbered_env_ports(
        prefix="PORT",
        max_count=SGLANG_DP_ATTENTION_ENV_PORT_MAX_COUNT,
    )
    if not env_ports:
        raise RuntimeError(
            "SGLang DP attention for multi-node rollout requires candidate "
            f"ports from PORT1..PORT{SGLANG_DP_ATTENTION_ENV_PORT_MAX_COUNT}."
        )

    skipped_avoided_ports = [port for port in env_ports if port in avoid_ports]
    candidate_ports = [port for port in env_ports if port not in avoid_ports]
    available_ports = []
    skipped_unavailable_ports = []
    for port in candidate_ports:
        if _is_port_available_on_address(address, port):
            available_ports.append(port)
        else:
            skipped_unavailable_ports.append(port)

    if skipped_avoided_ports or skipped_unavailable_ports:
        logger.info(
            "SGLang DP-attention env ports skipped on %s: reserved=%s unavailable=%s",
            address,
            skipped_avoided_ports,
            skipped_unavailable_ports,
        )

    start = replica_rank * SGLANG_DP_ATTENTION_AVAILABLE_PORT_COUNT
    selected_ports = available_ports[
        start : start + SGLANG_DP_ATTENTION_AVAILABLE_PORT_COUNT
    ]
    if len(selected_ports) != SGLANG_DP_ATTENTION_AVAILABLE_PORT_COUNT:
        raise RuntimeError(
            "SGLang DP attention for multi-node rollout needs "
            f"{SGLANG_DP_ATTENTION_AVAILABLE_PORT_COUNT} available PORT env "
            f"ports for replica {replica_rank} on {address}; env_ports={env_ports} "
            f"available_ports={available_ports} reserved={skipped_avoided_ports} "
            f"unavailable={skipped_unavailable_ports}"
        )
    return selected_ports


class SGLangHttpServer:
    """SGLang http server in single node, this is equivalent to launch server with command line:
    ```
    python -m sglang.launch_server --node-rank 0 --nnode 1 ...
    ```

    Args:
        config (DictConfig): full config.
        rollout_mode (RolloutMode): rollout mode.
        replica_rank (int): replica rank, a replica may contain multiple nodes.
        node_rank (int): node rank.
        nnodes (int): number of nodes.
        cuda_visible_devices (str): cuda visible devices.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        nnodes: int,
        cuda_visible_devices: str,
        base_gpu_id: int,
    ):
        print(f"SGLang http server: {rollout_mode=}, {replica_rank=}, {node_rank=}, {nnodes=}, {cuda_visible_devices=}")
        os.environ[visible_devices_keyword] = cuda_visible_devices

        self.config: RolloutConfig = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)
        max_position_embeddings = get_max_position_embeddings(self.model_config.hf_config)
        if self.config.max_model_len is None:
            self.config.max_model_len = max_position_embeddings
        else:
            if self.config.max_model_len > max_position_embeddings:
                raise ValueError(
                    f"max_model_len ({self.config.max_model_len}) should be less than or equal to "
                    f"max_position_embeddings ({max_position_embeddings})"
                )
        self.rollout_mode = rollout_mode
        self.workers = workers

        self.replica_rank = replica_rank
        self.node_rank = node_rank
        self.nnodes = nnodes
        self.base_gpu_id = base_gpu_id
        # model weights version, set by ServerAdapter when update weights.
        self.global_steps = None

        if self.rollout_mode != RolloutMode.HYBRID and self.config.load_format == "dummy":
            logger.warning(f"rollout mode is {self.rollout_mode}, load_format is dummy, set to auto")
            self.config.load_format = "auto"

        # used for http server
        self._server_address = ray.util.get_node_ip_address().strip("[]")
        self._server_port = None
        self.decoupled_spec_endpoint_infos: list[dict[str, Any]] = []

        # used for controlling sglang server profiler
        profiler_config = self.config.profiler
        tool_config = None
        if profiler_config is not None:
            if profiler_config.tool in ["torch", "npu"]:
                tool_config = omega_conf_to_dataclass((profiler_config.tool_config or {}).get(profiler_config.tool))
            else:
                logger.warning(f"agent loop only support torch and npu profiler, got {profiler_config.tool}")
                profiler_config = None
        self.profiler_controller = DistProfiler(self.replica_rank, config=profiler_config, tool_config=tool_config)

        # For multi-node, we need dist_init_addr so nodes can coordinate NCCL init.
        # For single-node, let SGLang handle port selection internally via nccl_port,
        # which also avoids port conflicts.
        self._master_address = None
        self._master_port = None
        self._master_sock = None
        self._master_socks = []
        self._dp_attention_available_ports = None
        if self.nnodes > 1 and self.node_rank == 0:
            self._master_address = self._server_address
            engine_kwargs = dict(self.config.get("engine_kwargs", {}).get("sglang", {}) or {})
            if engine_kwargs.get("enable_dp_attention", False):
                if "available_ports" not in [
                    f.name for f in dataclasses.fields(ServerArgs)
                ]:
                    raise RuntimeError(
                        "SGLang enable_dp_attention with multi-node rollout "
                        "requires ServerArgs.available_ports support."
                    )
                self._master_port, self._master_sock = get_free_port(
                    self._server_address,
                    with_alive_sock=True,
                )
                self._master_socks = [self._master_sock]
                if engine_kwargs.get("available_ports") is None:
                    self._dp_attention_available_ports = _select_sglang_dp_attention_available_ports(
                        self._server_address,
                        self.replica_rank,
                        avoid_ports={self._master_port},
                    )
                else:
                    self._dp_attention_available_ports = list(
                        engine_kwargs["available_ports"]
                    )
                logger.info(
                    "SGLangHttpServer, replica_rank: %s, DP-attention available_ports: %s",
                    self.replica_rank,
                    self._dp_attention_available_ports,
                )
            else:
                self._master_port, self._master_sock = get_free_port(
                    self._server_address,
                    with_alive_sock=True,
                )
                self._master_socks = [self._master_sock]
            logger.info(
                f"SGLangHttpServer, replica_rank: {self.replica_rank}, "
                f"master address: {self._master_address}, port: {self._master_port}"
            )

    def get_master_address(self):
        """Get master address and port for init NCCL process group."""
        return self._master_address, self._master_port

    def get_dp_attention_available_ports(self):
        """Get DP-attention available_ports selected on the rank-0 node."""
        return self._dp_attention_available_ports

    def get_server_address(self):
        """Get http server address and port."""
        assert self._server_port is not None, "http server is not launched, port is None"
        return self._server_address, self._server_port

    async def launch_server(
        self,
        master_address: str = None,
        master_port: int = None,
        dp_attention_available_ports: Optional[list[int]] = None,
        decoupled_spec_config: Optional[dict[str, Any]] = None,
    ):
        if self.nnodes > 1:
            if self.node_rank != 0:
                assert master_address and master_port, "non-master node should provide master address and port"
                self._master_address = master_address
                self._master_port = master_port
                self._dp_attention_available_ports = dp_attention_available_ports
            else:
                for sock in self._master_socks:
                    sock.close()
                self._master_socks = []
                self._master_sock = None

        engine_kwargs = dict(self.config.get("engine_kwargs", {}).get("sglang", {}) or {})
        attention_backend = engine_kwargs.pop("attention_backend", None)
        quantization = self.config.get("quantization", None)
        if quantization is not None:
            if quantization == "fp8":
                assert version.parse(sglang.__version__) >= version.parse("0.5.5"), (
                    "sglang>=0.5.5 is required for FP8 quantization"
                )
                FP8_BLOCK_QUANT_KWARGS = {
                    "activation_scheme": "dynamic",
                    "fmt": "e4m3",
                    "quant_method": "fp8",
                    "weight_block_size": [128, 128],
                }
                fp8_block_quant_kwargs = dict(FP8_BLOCK_QUANT_KWARGS)
            else:
                raise ValueError(f"Currently only support fp8 quantization, got: {quantization}")
        infer_tp = self.config.tensor_model_parallel_size * self.config.data_parallel_size
        args = {
            "model_path": self.model_config.local_path,
            "dtype": self.config.dtype,
            "mem_fraction_static": self.config.gpu_memory_utilization,
            "disable_cuda_graph": self.config.enforce_eager,
            "enable_memory_saver": True,
            "base_gpu_id": self.base_gpu_id,
            "gpu_id_step": 1,
            "tp_size": infer_tp,
            "dp_size": self.config.data_parallel_size,
            "ep_size": self.config.expert_parallel_size,
            "node_rank": self.node_rank,
            "load_format": self.config.load_format,
            "nnodes": self.nnodes,
            "trust_remote_code": self.model_config.trust_remote_code,
            "max_running_requests": self.config.get("max_num_seqs", None),
            "log_level": "error",
            "mm_attention_backend": "fa3",
            "attention_backend": attention_backend if attention_backend is not None else "fa3",
            "skip_tokenizer_init": self.config.skip_tokenizer_init,
            "skip_server_warmup": True,
            "quantization": quantization,
            "json_model_override_args": json.dumps({"quantization_config": fp8_block_quant_kwargs})
            if quantization == "fp8"
            else json.dumps({}),
            **engine_kwargs,
        }

        # Only set dist_init_addr for multi-node; for single-node, let SGLang
        # handle port selection internally via nccl_port to avoid conflicts.
        if self.nnodes > 1:
            dist_init_addr = (
                f"[{self._master_address}]:{self._master_port}"
                if is_valid_ipv6_address(self._master_address)
                else f"{self._master_address}:{self._master_port}"
            )
            args["dist_init_addr"] = dist_init_addr
            if (
                engine_kwargs.get("enable_dp_attention", False)
                and engine_kwargs.get("available_ports") is None
            ):
                if self._dp_attention_available_ports is None:
                    raise RuntimeError(
                        "SGLang DP attention requires available_ports selected "
                        "from the rank-0 node."
                    )
                args["available_ports"] = list(self._dp_attention_available_ports)

        if self.config.prometheus.enable:
            if self.config.prometheus.served_model_name:
                # Extract model name from path if it's a full path
                served_model_name = self.config.prometheus.served_model_name
                if "/" in served_model_name:
                    # If it's a full path, extract the last part as model name
                    served_model_name = served_model_name.split("/")[-1]
                args["served_model_name"] = served_model_name

            # start sglang metrics
            args["enable_metrics"] = True

        # enable_weights_cpu_backup is supported in sglang>=0.5.3
        if "enable_weights_cpu_backup" in [f.name for f in dataclasses.fields(ServerArgs)]:
            enable_weights_cpu_backup = True if self.rollout_mode == RolloutMode.COLOCATED else False
            args["enable_weights_cpu_backup"] = enable_weights_cpu_backup

        if self.config.enable_rollout_routing_replay:
            args.update({"enable_return_routed_experts": True})

        # mtp
        if self.config.mtp.enable and self.config.mtp.enable_rollout:
            # Enable weights CPU backup for sglang >= 0.5.6
            if sglang.__version__ < "0.5.6":
                raise ValueError(f"sglang version {sglang.__version__} is not supported for MTP rollout")

            args["speculative_algorithm"] = self.config.mtp.speculative_algorithm
            args["speculative_num_steps"] = self.config.mtp.speculative_num_steps
            args["speculative_eagle_topk"] = self.config.mtp.speculative_eagle_topk
            args["speculative_num_draft_tokens"] = self.config.mtp.speculative_num_draft_tokens

            args["enable_weights_cpu_backup"] = True
            args["enable_draft_weights_cpu_backup"] = True

        if decoupled_spec_config is not None:
            algorithm = decoupled_spec_config["algorithm"]
            speculative_num_steps = int(decoupled_spec_config["speculative_num_steps"])
            decoupled_spec_args = {
                "speculative_algorithm": algorithm,
                "speculative_num_steps": speculative_num_steps,
                "speculative_num_draft_tokens": speculative_num_steps + 1,
                "speculative_eagle_topk": 1,
                "decoupled_spec_rank_base": int(decoupled_spec_config["rank_base"]),
                "disable_overlap_schedule": True,
                "disable_radix_cache": True,
                "enable_mixed_chunk": False,
            }
            if algorithm == "DECOUPLED_DRAFT":
                decoupled_spec_args.update(
                    {
                        "chunked_prefill_size": -1,
                        "mamba_scheduler_strategy": "no_buffer",
                    }
                )
            spec_trace_dir = decoupled_spec_config.get("spec_trace_dir", None)
            if spec_trace_dir is not None:
                decoupled_spec_args["spec_trace_dir"] = spec_trace_dir
            args.update(decoupled_spec_args)

        # NOTE: We can't directly call SGLang's launch_server since it's not an async function.
        # https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py
        sglang.srt.entrypoints.engine._set_envs_and_config = _set_envs_and_config
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
        server_args = ServerArgs(**args)
        (
            self.tokenizer_manager,
            self.template_manager,
            _port_args,
            scheduler_init_result,
            subprocess_watchdog,
        ) = Engine._launch_subprocesses(
            server_args=server_args,
            init_tokenizer_manager_func=sglang.srt.entrypoints.engine.init_tokenizer_manager,
            run_scheduler_process_func=sglang.srt.entrypoints.engine.run_scheduler_process,
            run_detokenizer_process_func=sglang.srt.entrypoints.engine.run_detokenizer_process,
        )
        self.scheduler_infos = scheduler_init_result.scheduler_infos
        self.scheduler_info = self.scheduler_infos[0]
        self.decoupled_spec_endpoint_infos = [
            endpoint_info
            for scheduler_info in self.scheduler_infos
            for endpoint_info in scheduler_info.get("decoupled_spec_endpoint_infos", [])
        ]
        if self.tokenizer_manager is not None:
            self.tokenizer_manager._subprocess_watchdog = subprocess_watchdog

        # In multi-node cases, non-zero rank nodes should not launch http server.
        if self.node_rank > 0:
            return

        set_global_state(
            _GlobalState(
                tokenizer_manager=self.tokenizer_manager,
                template_manager=self.template_manager,
                scheduler_info=self.scheduler_info,
            )
        )
        app.is_single_tokenizer_mode = True

        # Set warmup_thread_{kw}args to avoid AttributeError in lifespan function
        app.server_args = server_args
        app.warmup_thread_kwargs = {"server_args": server_args}
        app.warmup_thread_args = (server_args, None, None)

        # Manually add Prometheus middleware before starting server
        # This ensures /metrics endpoint is available immediately
        if server_args.enable_metrics:
            from sglang.srt.utils.common import add_prometheus_middleware

            add_prometheus_middleware(app)

        self._server_port, self._server_task = await run_uvicorn(app, server_args, self._server_address)
        self.tokenizer_manager.server_status = ServerStatus.Up

    def get_decoupled_spec_endpoint_infos(self):
        return list(self.decoupled_spec_endpoint_infos)

    async def configure_decoupled_spec_peers(self, connect_endpoints: list[str]):
        assert self.node_rank == 0, "decoupled-spec peers must be configured through the entry server"
        assert self.tokenizer_manager is not None, "Tokenizer manager is not initialized"
        return await self.tokenizer_manager.configure_decoupled_spec_peers(list(connect_endpoints))

    async def wake_up(self):
        if self.node_rank != 0:
            return

        if self.rollout_mode == RolloutMode.HYBRID:
            # In hybrid mode, rollout is wake up in `update_weights`
            raise ValueError(f"wake_up not support rollout_mode {self.rollout_mode}")
        elif self.rollout_mode == RolloutMode.COLOCATED:
            # Directly call engine to wake up without sync weights.
            obj = ResumeMemoryOccupationReqInput(tags=["kv_cache", "weights"])
            await self.tokenizer_manager.resume_memory_occupation(obj, None)
            await self.tokenizer_manager.flush_cache()
        elif self.rollout_mode == RolloutMode.STANDALONE:
            # In standalone mode, resume kv_cache if free_cache_engine is enabled
            obj = ResumeMemoryOccupationReqInput(tags=["kv_cache"])
            await self.tokenizer_manager.resume_memory_occupation(obj, None)
            await self.tokenizer_manager.flush_cache()

    async def sleep(self):
        if self.node_rank != 0 or not self.config.free_cache_engine:
            return

        if self.rollout_mode == RolloutMode.HYBRID:
            obj = ReleaseMemoryOccupationReqInput(tags=["kv_cache", "weights"])
            await self.tokenizer_manager.release_memory_occupation(obj, None)
        elif self.rollout_mode == RolloutMode.COLOCATED:
            obj = ReleaseMemoryOccupationReqInput(tags=["kv_cache", "weights"])
            await self.tokenizer_manager.release_memory_occupation(obj, None)
        elif self.rollout_mode == RolloutMode.STANDALONE:
            # In standalone mode, resume kv_cache if free_cache_engine is enabled
            obj = ReleaseMemoryOccupationReqInput(tags=["kv_cache"])
            await self.tokenizer_manager.release_memory_occupation(obj, None)

    async def clear_kv_cache(self):
        if self.node_rank == 0:
            await self.tokenizer_manager.flush_cache()

    async def generate(
        self,
        prompt_ids: torch.Tensor,
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Generate sequence with token-in-token-out."""
        # TODO(@wuxibin): switch to `/generate` http endpoint once multi-modal support ready.
        max_possible_tokens = self.config.max_model_len - len(prompt_ids)

        if max_possible_tokens < 0:
            raise ValueError(
                f"Prompt length ({len(prompt_ids)}) exceeds the model's maximum context length "
                f"({self.config.max_model_len})."
            )

        if "max_new_tokens" in sampling_params:
            max_new_tokens = sampling_params.pop("max_new_tokens")
        elif "max_tokens" in sampling_params:
            # support vllm-style 'max_tokens' param
            max_new_tokens = sampling_params.pop("max_tokens")
        else:
            # Cap max_tokens by response_length to ensure tensor alignment,
            # and by remaining budget to prevent OOM in multi-turn rollouts.
            max_new_tokens = min(
                self.config.response_length, self.config.prompt_length + self.config.response_length - len(prompt_ids)
            )

        # Clamp max_new_tokens to the valid range [0, max_possible_tokens]
        max_new_tokens = max(0, min(max_new_tokens, max_possible_tokens))

        assert max_new_tokens <= max_possible_tokens, (
            f"max_new_tokens {max_new_tokens} exceeds available context space {max_possible_tokens}"
        )
        sampling_params["max_new_tokens"] = max_new_tokens
        return_logprob = sampling_params.pop("logprobs", False)

        request = {
            "rid": request_id,
            "input_ids": prompt_ids,
            "sampling_params": sampling_params,
            "return_logprob": return_logprob,
            "image_data": image_data,
            # TODO: support video input for sglang
            # video_data=video_data,
        }

        if self.config.enable_rollout_routing_replay:
            request.update({"return_routed_experts": True})

        generate_request = GenerateReqInput(**request)

        output = await self.tokenizer_manager.generate_request(generate_request, None).__anext__()
        meta_info = output["meta_info"]
        finish_reason = meta_info["finish_reason"]
        finish_reason = finish_reason["type"] if finish_reason else None
        if return_logprob:
            output_token_logprobs = meta_info["output_token_logprobs"]
            log_probs, token_ids = zip(
                *[(log_prob, token_ids) for log_prob, token_ids, _ in output_token_logprobs], strict=True
            )
        else:
            token_ids = output["output_ids"]
            log_probs = None

        routed_experts = None
        if self.config.enable_rollout_routing_replay:
            if self.config.skip_tokenizer_init:
                routed_experts = output.get("meta_info", {}).get("routed_experts", None)
            else:
                from sglang.srt.layers.moe.routed_experts_capturer import extract_routed_experts_from_meta_info

                hf_config = self.model_config.hf_config
                if not hasattr(hf_config, "num_hidden_layers") or not hasattr(hf_config, "num_experts_per_tok"):
                    raise AttributeError(
                        "enable_rollout_routing_replay is set, but hf_config is missing "
                        "'num_hidden_layers' or 'num_experts_per_tok'. This feature requires an MoE model "
                        "configuration that defines these attributes."
                    )
                routed_experts = extract_routed_experts_from_meta_info(output).reshape(
                    -1, hf_config.num_hidden_layers, hf_config.num_experts_per_tok
                )

        sglang_meta_info = {
            key: value
            for key, value in meta_info.items()
            if key.startswith("spec_")
            or key
            in {
                "prompt_tokens",
                "completion_tokens",
                "cached_tokens",
                "reasoning_tokens",
                "e2e_latency",
            }
        }
        sglang_meta_info["finish_reason"] = finish_reason

        return TokenOutput(
            token_ids=token_ids,
            log_probs=log_probs,
            routed_experts=routed_experts,
            stop_reason=finish_reason,
            extra_fields={"global_steps": self.global_steps, "sglang_meta_info": sglang_meta_info},
        )

    async def set_global_steps(self, global_steps: int):
        """Set the global steps of the model weights."""
        self.global_steps = global_steps

    async def abort_all_requests(self):
        if self.node_rank != 0:
            return
        await self.tokenizer_manager.pause_generation(PauseGenerationReqInput(mode="abort"))

    async def resume_generation(self):
        if self.node_rank != 0:
            return
        await self.tokenizer_manager.continue_generation(ContinueGenerationReqInput())

    async def start_profile(self, **kwargs):
        if (
            self.profiler_controller.check_enable()
            and self.profiler_controller.check_this_rank()
            and self.profiler_controller.is_discrete_mode()
        ):
            profile_args = build_sglang_profiler_args(
                self.profiler_controller.config, self.profiler_controller.tool_config, self.replica_rank
            )
            await self.tokenizer_manager.start_profile(**profile_args)

    async def stop_profile(self):
        if (
            self.profiler_controller.check_enable()
            and self.profiler_controller.check_this_rank()
            and self.profiler_controller.is_discrete_mode()
        ):
            await self.tokenizer_manager.stop_profile()


class SGLangReplica(RolloutReplica):
    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        self.server_class = ray.remote(SGLangHttpServer)
        self.decoupled_spec_config: Optional[dict[str, Any]] = None

    def set_decoupled_spec_config(self, decoupled_spec_config: Optional[dict[str, Any]]):
        self.decoupled_spec_config = decoupled_spec_config

    async def prepare_server_actors(self, server_name_prefix: Optional[str] = None):
        """Create SGLang server actors in each node without launching SGLang."""
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )
        if self.servers:
            return None

        server_name_prefix = server_name_prefix or ("sglang_server_reward" if self.is_reward_model else "sglang_server")
        # get (node_id, CUDA_VISIBLE_DEVICES) of all workers
        worker_infos = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda self: (ray.get_runtime_context().get_node_id(), os.environ[visible_devices_keyword])
                )
                for worker in self.workers
            ]
        )
        worker_cuda_visible_devices = [worker_info[1] for worker_info in worker_infos]
        worker_node_ids = [worker_info[0] for worker_info in worker_infos]
        entry_runtime_info = {"node_id": worker_node_ids[0]}
        base_gpu_id = 0
        infer_tp = self.config.tensor_model_parallel_size * self.config.data_parallel_size
        replica_world_size = infer_tp * self.config.pipeline_model_parallel_size
        if os.environ.get(f"RAY_EXPERIMENTAL_NOSET_{visible_devices_keyword}", None):
            logger.warning(f"RAY_EXPERIMENTAL_NOSET_{visible_devices_keyword} is set True!")
            base_gpu_id = (0 + self.replica_rank * replica_world_size) % self.gpus_per_node
        # create server actor in each node with node affinity and cuda visible devices
        for node_rank in range(self.nnodes):
            workers = self.workers[
                node_rank * self.gpus_per_replica_node : (node_rank + 1) * self.gpus_per_replica_node
            ]
            node_cuda_visible_devices_set = worker_cuda_visible_devices[
                node_rank * self.gpus_per_replica_node : (node_rank + 1) * self.gpus_per_replica_node
            ]
            node_cuda_visible_devices = ",".join(
                map(
                    str,
                    sorted(
                        set(
                            int(device)
                            for worker_devices_set in node_cuda_visible_devices_set
                            for device in worker_devices_set.split(",")
                            if device.strip()
                        )
                    ),
                )
            )

            node_id = worker_node_ids[node_rank * self.gpus_per_replica_node]
            name = f"{server_name_prefix}_{self.replica_rank}_{node_rank}"

            server = self.server_class.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                runtime_env={"env_vars": {f"RAY_EXPERIMENTAL_NOSET_{visible_devices_keyword}": "1"}},
                name=name,
                max_concurrency=self.max_concurrency,
            ).remote(
                config=self.config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                workers=workers,
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                nnodes=self.nnodes,
                cuda_visible_devices=node_cuda_visible_devices,
                base_gpu_id=base_gpu_id,
            )
            self.servers.append(server)
        return entry_runtime_info

    async def launch_prepared_servers(self, decoupled_spec_config: Optional[dict[str, Any]] = None):
        """Launch already-created server actors."""
        await self.prepare_server_actors()
        decoupled_spec_config = decoupled_spec_config or self.decoupled_spec_config

        # launch http server in each node
        master_address, master_port = None, None
        dp_attention_available_ports = None
        if self.nnodes > 1:
            master_address, master_port = await self.servers[0].get_master_address.remote()
            dp_attention_available_ports = (
                await self.servers[0].get_dp_attention_available_ports.remote()
            )
        await asyncio.gather(
            *[
                server.launch_server.remote(
                    master_address=master_address,
                    master_port=master_port,
                    dp_attention_available_ports=dp_attention_available_ports,
                    decoupled_spec_config=decoupled_spec_config,
                )
                for server in self.servers
            ]
        )

        # get http server address from first server
        server_address, server_port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = (
            f"[{server_address}]:{server_port}"
            if is_valid_ipv6_address(server_address)
            else f"{server_address}:{server_port}"
        )

    async def get_decoupled_spec_endpoint_infos(self):
        """Collect published decoupled-spec bind endpoints from all server actors."""
        await self.prepare_server_actors()
        endpoint_infos_by_server = await asyncio.gather(
            *[server.get_decoupled_spec_endpoint_infos.remote() for server in self.servers]
        )
        return [info for endpoint_infos in endpoint_infos_by_server for info in endpoint_infos]

    async def configure_decoupled_spec_peers(self, connect_endpoints: list[str]):
        """Configure decoupled-spec peer endpoints through the entry server."""
        await self.prepare_server_actors()
        return await self.servers[0].configure_decoupled_spec_peers.remote(list(connect_endpoints))

    async def launch_servers(self):
        """Launch http server in each node."""
        await self.prepare_server_actors()
        await self.launch_prepared_servers()
