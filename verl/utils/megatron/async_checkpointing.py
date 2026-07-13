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

from typing import Any

_async_calls: Any | None = None


def get_async_calls():
    """Return the MCore async checkpoint queue across MCore API layouts."""
    try:
        from megatron.core.dist_checkpointing.strategies.base import async_calls

        return async_calls
    except ImportError:
        pass

    global _async_calls
    if _async_calls is None:
        from megatron.core.dist_checkpointing.strategies.async_utils import AsyncCallsQueue

        _async_calls = AsyncCallsQueue()
    return _async_calls


def schedule_async_request(async_save_request):
    return get_async_calls().schedule_async_request(async_save_request)


def maybe_finalize_async_calls(blocking=False):
    return get_async_calls().maybe_finalize_async_calls(blocking=blocking)
