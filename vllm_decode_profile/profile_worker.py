"""Rank-filtered, decode-only profiler trigger for vLLM-Ascend.

This worker changes profiling control only. Model execution, KV-cache behavior,
and all baseline vLLM-Ascend kernels still come from ``NPUWorker`` unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

from vllm_ascend.worker.worker import NPUWorker

from vllm_decode_profile.profile_common import single_token_decode_info
from vllm_decode_profile.profile_env import (
    env_bool,
    env_non_negative_int,
    env_optional_positive_int,
)


LOGGER = logging.getLogger(__name__)
EXPECTED_BATCH_ENV = "VLLM_PROFILE_EXPECTED_BATCH_SIZE"
TRIGGER_BATCH_ENV = "VLLM_PROFILE_TRIGGER_BATCH_SIZE"
LEGACY_PROFILE_RANK_ENV = "VLLM_PROFILE_GLOBAL_RANK"
PROFILE_DP_RANK_ENV = "VLLM_PROFILE_TARGET_DP_RANK"
PROFILE_TP_RANK_ENV = "VLLM_PROFILE_TARGET_TP_RANK"
MAX_DECODE_STEPS_ENV = "VLLM_PROFILE_MAX_DECODE_STEPS"
STOP_ON_PHASE_CHANGE_ENV = "VLLM_PROFILE_STOP_ON_PHASE_CHANGE"


class DecodeOnlyRankFilteredNPUWorker(NPUWorker):
    """Profile pure decode on one selected DP/TP worker.

    ``NPUWorker.rank`` is scoped to the executor inside a DP EngineCore. With
    DP>1 and TP=1 every DP worker therefore has rank 0, so it cannot be used to
    distinguish DP ranks. ``ParallelConfig.data_parallel_rank`` is the stable
    global DP selector for both local and headless EngineCores.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self._profile_dp_rank = int(self.parallel_config.data_parallel_rank)
        tp_size = int(self.parallel_config.tensor_parallel_size)
        if tp_size <= 0:
            raise RuntimeError(
                "tensor_parallel_size must be positive, got "
                f"{self.parallel_config.tensor_parallel_size!r}"
            )
        # Worker ranks are local to one DP EngineCore. Modulo also keeps this
        # selector correct if PP is introduced later.
        self._profile_tp_rank = int(self.rank) % tp_size

        legacy_target_rank = env_non_negative_int(
            LEGACY_PROFILE_RANK_ENV,
            0,
        )
        self._profile_target_dp_rank = env_non_negative_int(
            PROFILE_DP_RANK_ENV,
            0,
        )
        self._profile_target_tp_rank = env_non_negative_int(
            PROFILE_TP_RANK_ENV,
            legacy_target_rank,
        )
        dp_size = int(self.parallel_config.data_parallel_size)
        if self._profile_target_dp_rank >= dp_size:
            raise RuntimeError(
                f"{PROFILE_DP_RANK_ENV} must be in [0, {dp_size}), got "
                f"{self._profile_target_dp_rank}"
            )
        if self._profile_target_tp_rank >= tp_size:
            raise RuntimeError(
                f"{PROFILE_TP_RANK_ENV} must be in [0, {tp_size}), got "
                f"{self._profile_target_tp_rank}"
            )

        self._profile_expected_batch_size = env_optional_positive_int(
            TRIGGER_BATCH_ENV,
            EXPECTED_BATCH_ENV,
        )
        self._profile_max_decode_steps = env_non_negative_int(
            MAX_DECODE_STEPS_ENV,
            0,
        )
        self._profile_stop_on_phase_change = env_bool(
            STOP_ON_PHASE_CHANGE_ENV,
            True,
        )
        self._decode_profile_armed = False
        self._decode_profile_started = False
        self._decode_profile_finished = False
        self._profiled_decode_steps = 0
        self._decode_profile_prefix: str | None = None

    def _is_target_rank(self) -> bool:
        return (
            self._profile_dp_rank == self._profile_target_dp_rank
            and self._profile_tp_rank == self._profile_target_tp_rank
        )

    def _is_profile_trigger_decode(self, scheduler_output: Any) -> bool:
        return single_token_decode_info(
            scheduler_output,
            expected_batch_size=self._profile_expected_batch_size,
        ) is not None

    def _stop_running_profile(self, reason: str) -> None:
        if not self._decode_profile_started:
            return
        super().profile(is_start=False)
        self._decode_profile_started = False
        self._decode_profile_armed = False
        self._decode_profile_finished = True
        self._decode_profile_prefix = None
        LOGGER.warning(
            "VLLM_BASELINE_PROFILE_STOPPED dp_rank=%d tp_rank=%d "
            "worker_rank=%d decode_steps=%d reason=%s",
            self._profile_dp_rank,
            self._profile_tp_rank,
            self.rank,
            self._profiled_decode_steps,
            reason,
        )

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
                    "Decode-only profiler is already armed or running on "
                    "dp_rank=%d tp_rank=%d worker_rank=%d.",
                    self._profile_dp_rank,
                    self._profile_tp_rank,
                    self.rank,
                )
                return
            self._decode_profile_armed = True
            self._decode_profile_finished = False
            self._profiled_decode_steps = 0
            self._decode_profile_prefix = profile_prefix
            expected_batch = (
                str(self._profile_expected_batch_size)
                if self._profile_expected_batch_size is not None
                else "any"
            )
            max_steps = (
                str(self._profile_max_decode_steps)
                if self._profile_max_decode_steps
                else "unlimited"
            )
            LOGGER.warning(
                "VLLM_BASELINE_PROFILE_ARMED dp_rank=%d tp_rank=%d "
                "worker_rank=%d expected_local_batch=%s max_decode_steps=%s; "
                "prefill will not be recorded",
                self._profile_dp_rank,
                self._profile_tp_rank,
                self.rank,
                expected_batch,
                max_steps,
            )
            return

        was_armed = self._decode_profile_armed
        self._decode_profile_armed = False
        if self._decode_profile_started:
            self._stop_running_profile(reason="stop_requested")
        elif was_armed:
            LOGGER.warning(
                "VLLM_BASELINE_PROFILE_NOT_STARTED dp_rank=%d tp_rank=%d "
                "worker_rank=%d: no matching single-token decode step was "
                "observed",
                self._profile_dp_rank,
                self._profile_tp_rank,
                self.rank,
            )
        elif self._decode_profile_finished:
            LOGGER.info(
                "Decode-only profiler on dp_rank=%d tp_rank=%d was already "
                "stopped automatically.",
                self._profile_dp_rank,
                self._profile_tp_rank,
            )
        self._decode_profile_prefix = None

    def execute_model(self, scheduler_output: Any) -> Any:
        is_target = self._is_target_rank()
        is_trigger_decode = (
            is_target
            and self._is_profile_trigger_decode(scheduler_output)
        )

        if (
            is_target
            and self._decode_profile_started
            and self._profile_stop_on_phase_change
            and not is_trigger_decode
        ):
            # Stop before executing a mixed-prefill, partial-batch, or dummy EP
            # step so the recorded device range remains decode-only.
            self._stop_running_profile(reason="decode_phase_changed")

        if (
            is_target
            and self._decode_profile_armed
            and not self._decode_profile_started
            and is_trigger_decode
        ):
            profile_prefix = self._decode_profile_prefix or (
                f"vllm_decode_dp{self._profile_dp_rank}_"
                f"tp{self._profile_tp_rank}"
            )
            LOGGER.warning(
                "VLLM_BASELINE_PROFILE_STARTED dp_rank=%d tp_rank=%d "
                "worker_rank=%d local_batch=%d phase=decode",
                self._profile_dp_rank,
                self._profile_tp_rank,
                self.rank,
                len(scheduler_output.scheduled_cached_reqs.req_ids),
            )
            super().profile(
                is_start=True,
                profile_prefix=profile_prefix,
            )
            self._decode_profile_started = True
            self._decode_profile_armed = False

        output = super().execute_model(scheduler_output)

        if (
            is_target
            and self._decode_profile_started
            and is_trigger_decode
        ):
            self._profiled_decode_steps += 1
            if (
                self._profile_max_decode_steps
                and self._profiled_decode_steps
                >= self._profile_max_decode_steps
            ):
                self._stop_running_profile(reason="max_decode_steps")

        return output
