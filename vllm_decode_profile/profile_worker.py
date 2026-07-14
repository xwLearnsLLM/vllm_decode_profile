"""Rank-filtered, decode-only profiler trigger for vLLM-Ascend.

This worker changes profiling control only. Model execution, KV-cache behavior,
and all baseline vLLM-Ascend kernels still come from ``NPUWorker`` unchanged.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from vllm_ascend.worker.worker import NPUWorker

from vllm_decode_profile.profile_common import single_token_decode_info


LOGGER = logging.getLogger(__name__)
EXPECTED_BATCH_ENV = "VLLM_PROFILE_EXPECTED_BATCH_SIZE"
PROFILE_RANK_ENV = "VLLM_PROFILE_GLOBAL_RANK"


def _required_non_negative_int(name: str) -> int:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(f"{name} must be set by profile_vllm.py")
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {value!r}") from exc
    if parsed < 0:
        raise RuntimeError(f"{name} must be non-negative, got {parsed}")
    return parsed


class DecodeOnlyRankFilteredNPUWorker(NPUWorker):
    """Start the native profiler on one rank at the first full-batch decode."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._profile_expected_batch_size = _required_non_negative_int(
            EXPECTED_BATCH_ENV
        )
        if self._profile_expected_batch_size == 0:
            raise RuntimeError(f"{EXPECTED_BATCH_ENV} must be greater than zero")
        self._profile_target_rank = _required_non_negative_int(PROFILE_RANK_ENV)
        self._decode_profile_armed = False
        self._decode_profile_started = False
        self._decode_profile_prefix: str | None = None

    def _is_target_rank(self) -> bool:
        return int(self.rank) == self._profile_target_rank

    def _is_full_batch_single_token_decode(self, scheduler_output: Any) -> bool:
        return single_token_decode_info(
            scheduler_output,
            expected_batch_size=self._profile_expected_batch_size,
        ) is not None

    def profile(
        self,
        is_start: bool = True,
        profile_prefix: str | None = None,
    ) -> None:
        # start_profile/stop_profile are broadcast to every TP worker. Non-target
        # ranks intentionally never create a torch_npu profiler.
        if not self._is_target_rank():
            return

        if is_start:
            if self._decode_profile_armed or self._decode_profile_started:
                LOGGER.warning(
                    "Decode-only profiler is already armed or running on rank %d.",
                    self.rank,
                )
                return
            self._decode_profile_armed = True
            self._decode_profile_prefix = profile_prefix
            LOGGER.warning(
                "VLLM_BASELINE_PROFILE_ARMED rank=%d expected_batch=%d; "
                "prefill will not be recorded.",
                self.rank,
                self._profile_expected_batch_size,
            )
            return

        was_armed = self._decode_profile_armed
        self._decode_profile_armed = False
        if self._decode_profile_started:
            super().profile(is_start=False)
            self._decode_profile_started = False
            LOGGER.warning("VLLM_BASELINE_PROFILE_STOPPED rank=%d", self.rank)
        elif was_armed:
            LOGGER.warning(
                "VLLM_BASELINE_PROFILE_NOT_STARTED rank=%d: no full-batch "
                "single-token decode step was observed.",
                self.rank,
            )
        self._decode_profile_prefix = None

    def execute_model(self, scheduler_output: Any) -> Any:
        if (
            self._is_target_rank()
            and self._decode_profile_armed
            and not self._decode_profile_started
            and self._is_full_batch_single_token_decode(scheduler_output)
        ):
            LOGGER.warning(
                "VLLM_BASELINE_PROFILE_STARTED rank=%d batch=%d phase=decode",
                self.rank,
                self._profile_expected_batch_size,
            )
            super().profile(
                is_start=True,
                profile_prefix=self._decode_profile_prefix,
            )
            self._decode_profile_started = True
            self._decode_profile_armed = False

        return super().execute_model(scheduler_output)
