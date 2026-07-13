import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any


TIMELINE_FILENAME = "timeline.jsonl"


def _config_get(config: Any, key: str) -> Any:
    if config is None:
        return None
    if hasattr(config, "get"):
        return config.get(key, None)
    return getattr(config, key, None)


def _root_rollout_config(config: Any) -> Any:
    actor_rollout_ref = _config_get(config, "actor_rollout_ref")
    if actor_rollout_ref is not None:
        return _config_get(actor_rollout_ref, "rollout")
    return _config_get(config, "rollout")


def _rollout_trace_dir(config: Any) -> str | None:
    rollout_config = _root_rollout_config(config)
    trace_config = _config_get(rollout_config, "trace")
    trace_dir = _config_get(trace_config, "trace_dir")
    return str(trace_dir) if trace_dir else None


def _draft_output_dir(config: Any) -> str | None:
    draft_config = _config_get(config, "draft")
    output_dir = _config_get(draft_config, "output_dir")
    return str(output_dir) if output_dir else None


def get_timeline_output_dir(config_or_path: Any) -> str | None:
    if config_or_path is None:
        return None
    if isinstance(config_or_path, (str, os.PathLike)):
        output_dir = os.fspath(config_or_path)
        return output_dir if output_dir else None

    return _rollout_trace_dir(config_or_path) or _draft_output_dir(config_or_path)


def _timestamp(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat().replace("+00:00", "Z")


def append_timeline_event(
    config_or_path: Any,
    *,
    stage: str,
    event: str,
    global_step: int | None = None,
    epoch: int | None = None,
    validate: bool | None = None,
    **metadata: Any,
) -> None:
    output_dir = get_timeline_output_dir(config_or_path)
    if not output_dir:
        return

    ts = time.time()
    record = {
        "timestamp": _timestamp(ts),
        "time_unix": ts,
        "stage": stage,
        "event": event,
    }
    if global_step is not None:
        record["global_step"] = global_step
    if epoch is not None:
        record["epoch"] = epoch
    if validate is not None:
        record["validate"] = bool(validate)
    if metadata:
        record["metadata"] = metadata

    os.makedirs(output_dir, exist_ok=True)
    timeline_path = os.path.join(output_dir, TIMELINE_FILENAME)
    with open(timeline_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")


@contextmanager
def timeline_span(
    config_or_path: Any,
    stage: str,
    *,
    global_step: int | None = None,
    epoch: int | None = None,
    validate: bool | None = None,
    **metadata: Any,
):
    start = time.time()
    append_timeline_event(
        config_or_path,
        stage=stage,
        event="start",
        global_step=global_step,
        epoch=epoch,
        validate=validate,
        **metadata,
    )
    try:
        yield
    except Exception as exc:
        append_timeline_event(
            config_or_path,
            stage=stage,
            event="error",
            global_step=global_step,
            epoch=epoch,
            validate=validate,
            duration_seconds=time.time() - start,
            error_type=type(exc).__name__,
            error_message=str(exc),
            **metadata,
        )
        raise
    else:
        append_timeline_event(
            config_or_path,
            stage=stage,
            event="end",
            global_step=global_step,
            epoch=epoch,
            validate=validate,
            duration_seconds=time.time() - start,
            **metadata,
        )
