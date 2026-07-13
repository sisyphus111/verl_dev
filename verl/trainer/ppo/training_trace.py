# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import csv
import json
import os
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.utils.timeline import get_timeline_output_dir


def _trace_step_output_dir(config: Any, global_step: int) -> str | None:
    base_output_dir = get_timeline_output_dir(config)
    if not base_output_dir:
        return None

    output_dir = os.path.join(base_output_dir, f"global_step_{global_step}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _to_jsonable(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return _to_jsonable(value.item())
        return _to_jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _csv_value(value: Any) -> Any:
    value = _to_jsonable(value)
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _sequence_value(values: Any, index: int, default=None) -> Any:
    if values is None:
        return default
    try:
        if index >= len(values):
            return default
        return values[index]
    except TypeError:
        return default


def _masked_tensor_row_stats(tensor: torch.Tensor, mask: torch.Tensor) -> dict[str, list[Any]]:
    tensor = tensor.float()
    mask = mask.bool()
    lengths = mask.sum(dim=-1)
    mask_float = mask.to(dtype=tensor.dtype)
    sums = (tensor * mask_float).sum(dim=-1)
    means = sums / lengths.clamp(min=1).to(dtype=tensor.dtype)
    mins = tensor.masked_fill(~mask, float("inf")).min(dim=-1).values
    maxes = tensor.masked_fill(~mask, float("-inf")).max(dim=-1).values

    lengths_list = lengths.detach().cpu().tolist()
    raw_stats = {
        "mean": means.detach().cpu().tolist(),
        "sum": sums.detach().cpu().tolist(),
        "min": mins.detach().cpu().tolist(),
        "max": maxes.detach().cpu().tolist(),
    }
    stats: dict[str, list[Any]] = {key: [] for key in raw_stats}
    for i, length in enumerate(lengths_list):
        for key, values in raw_stats.items():
            stats[key].append(None if length == 0 else _to_jsonable(values[i]))
    return stats


def _dump_training_request_signals_trace(
    *,
    batch: DataProto,
    global_step: int,
    epoch: int,
    reward_extra_infos_dict: dict,
    output_dir: str,
) -> None:
    response_mask = batch.batch["response_mask"].bool()
    max_response_length = batch.batch["responses"].shape[-1]
    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()

    prompt_lengths = prompt_mask.sum(dim=-1).detach().cpu().tolist()
    response_lengths = response_mask.sum(dim=-1).detach().cpu().tolist()
    sequence_scores = batch.batch["token_level_scores"].sum(dim=-1).detach().float().cpu().tolist()
    sequence_rewards = batch.batch["token_level_rewards"].sum(dim=-1).detach().float().cpu().tolist()
    advantage_stats = _masked_tensor_row_stats(batch.batch["advantages"], response_mask)
    return_stats = _masked_tensor_row_stats(batch.batch["returns"], response_mask)
    value_stats = _masked_tensor_row_stats(batch.batch["values"], response_mask) if "values" in batch.batch else None
    optional_tensor_stats = {
        output_name: _masked_tensor_row_stats(batch.batch[tensor_name], response_mask)
        for tensor_name, output_name in (
            ("old_log_probs", "old_log_prob"),
            ("ref_log_prob", "ref_log_prob"),
            ("rollout_log_probs", "rollout_log_prob"),
            ("rollout_is_weights", "rollout_is_weight"),
        )
        if tensor_name in batch.batch
    }

    num_rows = batch.batch.batch_size[0]
    uid_values = batch.non_tensor_batch.get("uid")
    group_scores: dict[str, list[float]] = defaultdict(list)
    group_ids: list[Any] = []
    for i in range(num_rows):
        group_id = _to_jsonable(_sequence_value(uid_values, i, None))
        group_ids.append(group_id)
        if group_id is not None:
            group_key = json.dumps(group_id, ensure_ascii=False, sort_keys=True)
            group_scores[group_key].append(float(sequence_rewards[i]))

    csv_path = os.path.join(output_dir, "training_signals.csv")
    jsonl_path = os.path.join(output_dir, "training_signals.jsonl")
    csv_fields = [
        "global_step",
        "epoch",
        "row_index",
        "original_dataset_file_index",
        "original_dataset_row_index",
        "grpo_group_id",
        "grpo_group_size",
        "grpo_group_score_mean",
        "grpo_group_score_std",
        "prompt_length",
        "response_length",
        "response_aborted",
        "finish_reason",
        "e2e_latency",
        "sequence_score",
        "sequence_reward",
        "advantage_mean",
        "advantage_sum",
        "advantage_min",
        "advantage_max",
        "return_mean",
        "return_sum",
        "return_min",
        "return_max",
        "value_mean",
        "value_sum",
        "value_min",
        "value_max",
        "old_log_prob_mean",
        "old_log_prob_sum",
        "old_log_prob_min",
        "old_log_prob_max",
        "ref_log_prob_mean",
        "ref_log_prob_sum",
        "ref_log_prob_min",
        "ref_log_prob_max",
        "rollout_log_prob_mean",
        "rollout_log_prob_sum",
        "rollout_log_prob_min",
        "rollout_log_prob_max",
        "rollout_is_weight_mean",
        "rollout_is_weight_sum",
        "rollout_is_weight_min",
        "rollout_is_weight_max",
        "ground_truth",
        "reward_extra_info",
    ]
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0

    with open(csv_path, "a", newline="", encoding="utf-8") as csv_file, open(
        jsonl_path, "a", encoding="utf-8"
    ) as jsonl_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        if write_header:
            writer.writeheader()

        for i in range(num_rows):
            sglang_meta_info = _to_jsonable(_sequence_value(batch.non_tensor_batch.get("sglang_meta_info"), i, {}))
            if not isinstance(sglang_meta_info, dict):
                sglang_meta_info = {}

            reward_model = _to_jsonable(_sequence_value(batch.non_tensor_batch.get("reward_model"), i, {}))
            ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None

            reward_extra_info = {}
            for key, values in reward_extra_infos_dict.items():
                try:
                    has_row_value = values is not None and len(values) == num_rows
                except TypeError:
                    has_row_value = False
                if has_row_value:
                    reward_extra_info[key] = _to_jsonable(values[i])

            group_id = group_ids[i]
            group_key = json.dumps(group_id, ensure_ascii=False, sort_keys=True) if group_id is not None else None
            current_group_scores = group_scores.get(group_key, []) if group_key is not None else []
            if current_group_scores:
                group_size = len(current_group_scores)
                group_score_mean = float(np.mean(current_group_scores))
                group_score_std = float(np.std(current_group_scores, ddof=1)) if group_size > 1 else 0.0
            else:
                group_size = None
                group_score_mean = None
                group_score_std = None

            record = {
                "global_step": global_step,
                "epoch": epoch,
                "row_index": i,
                "original_dataset_file_index": _to_jsonable(
                    _sequence_value(batch.non_tensor_batch.get("original_dataset_file_index"), i, None)
                ),
                "original_dataset_row_index": _to_jsonable(
                    _sequence_value(batch.non_tensor_batch.get("original_dataset_row_index"), i, None)
                ),
                "grpo_group_id": group_id,
                "grpo_group_size": group_size,
                "grpo_group_score_mean": _to_jsonable(group_score_mean),
                "grpo_group_score_std": _to_jsonable(group_score_std),
                "prompt_length": int(prompt_lengths[i]),
                "response_length": int(response_lengths[i]),
                "response_aborted": bool(response_lengths[i] == 0),
                "finish_reason": sglang_meta_info.get("finish_reason"),
                "e2e_latency": sglang_meta_info.get("e2e_latency"),
                "sequence_score": _to_jsonable(sequence_scores[i]),
                "sequence_reward": _to_jsonable(sequence_rewards[i]),
                "advantage_mean": advantage_stats["mean"][i],
                "advantage_sum": advantage_stats["sum"][i],
                "advantage_min": advantage_stats["min"][i],
                "advantage_max": advantage_stats["max"][i],
                "return_mean": return_stats["mean"][i],
                "return_sum": return_stats["sum"][i],
                "return_min": return_stats["min"][i],
                "return_max": return_stats["max"][i],
                "value_mean": value_stats["mean"][i] if value_stats is not None else None,
                "value_sum": value_stats["sum"][i] if value_stats is not None else None,
                "value_min": value_stats["min"][i] if value_stats is not None else None,
                "value_max": value_stats["max"][i] if value_stats is not None else None,
                **{
                    f"{name}_{stat_name}": stats[stat_name][i]
                    for name, stats in optional_tensor_stats.items()
                    for stat_name in ("mean", "sum", "min", "max")
                },
                "ground_truth": ground_truth,
                "reward_extra_info": reward_extra_info,
                "sglang_meta_info": sglang_meta_info,
            }
            jsonl_file.write(json.dumps(_to_jsonable(record), ensure_ascii=False) + "\n")
            writer.writerow({field: _csv_value(record.get(field)) for field in csv_fields})


def _dump_training_step_metrics_trace(
    *,
    metrics: dict[str, Any],
    global_step: int,
    epoch: int,
    output_dir: str,
) -> None:
    jsonl_path = os.path.join(output_dir, "training_step_metrics.jsonl")
    csv_path = os.path.join(output_dir, "training_step_metrics.csv")
    record = {
        "global_step": global_step,
        "epoch": epoch,
        "metrics": _to_jsonable(metrics),
    }
    with open(jsonl_path, "a", encoding="utf-8") as jsonl_file:
        jsonl_file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    csv_fields = ["global_step", "epoch", "metric_name", "metric_value"]
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        if write_header:
            writer.writeheader()
        for name in sorted(metrics):
            writer.writerow(
                {
                    "global_step": global_step,
                    "epoch": epoch,
                    "metric_name": name,
                    "metric_value": _csv_value(metrics[name]),
                }
            )


def dump_training_trace(
    *,
    config: Any,
    batch: DataProto,
    metrics: dict[str, Any],
    global_step: int,
    epoch: int,
    reward_extra_infos_dict: dict,
) -> None:
    output_dir = _trace_step_output_dir(config, global_step)
    if not output_dir:
        return

    _dump_training_request_signals_trace(
        batch=batch,
        global_step=global_step,
        epoch=epoch,
        reward_extra_infos_dict=reward_extra_infos_dict,
        output_dir=output_dir,
    )
    _dump_training_step_metrics_trace(metrics=metrics, global_step=global_step, epoch=epoch, output_dir=output_dir)
