"""EngineCore decode-step statistics for the vLLM-Ascend baseline."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from vllm.v1.core.sched.scheduler import Scheduler

from vllm_decode_profile.profile_common import single_token_decode_info
from vllm_decode_profile.profile_env import (
    env_non_negative_float,
    env_non_negative_int,
)


PROFILE_DP_RANK_ENV = "VLLM_PROFILE_TARGET_DP_RANK"
DECODE_LOG_DP_RANK_ENV = "VLLM_DECODE_LOG_DP_RANK"
DECODE_LOG_FLUSH_STEPS_ENV = "VLLM_DECODE_LOG_FLUSH_STEPS"
DECODE_LOG_FLUSH_SECONDS_ENV = "VLLM_DECODE_LOG_FLUSH_SECONDS"


@dataclass(frozen=True)
class _PendingDecodeStep:
    start_time: float
    batch_size: int
    num_tokens: int
    seq_lens: tuple[int, ...]
    kv_used_blocks: int
    kv_total_blocks: int


@dataclass(frozen=True)
class _CompletedDecodeStep:
    step: int
    batch_size: int
    num_tokens: int
    tpot_ms: float
    seq_lens: tuple[int, ...]
    kv_used_blocks: int
    kv_total_blocks: int


class DecodeStepLoggingScheduler(Scheduler):
    """Measure pure decode steps on one selected DP EngineCore.

    A Scheduler belongs to a DP EngineCore rather than an individual TP
    worker. Selecting one DP rank is therefore sufficient for DP=32, TP=1.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._decode_log_dp_rank = int(
            self.parallel_config.data_parallel_rank
        )
        default_target_dp_rank = env_non_negative_int(
            PROFILE_DP_RANK_ENV,
            0,
        )
        self._decode_log_target_dp_rank = env_non_negative_int(
            DECODE_LOG_DP_RANK_ENV,
            default_target_dp_rank,
        )
        dp_size = int(self.parallel_config.data_parallel_size)
        if self._decode_log_target_dp_rank >= dp_size:
            raise RuntimeError(
                f"{DECODE_LOG_DP_RANK_ENV} must be in [0, {dp_size}), got "
                f"{self._decode_log_target_dp_rank}"
            )
        self._decode_log_enabled = (
            self._decode_log_dp_rank == self._decode_log_target_dp_rank
        )
        self._decode_log_flush_steps = env_non_negative_int(
            DECODE_LOG_FLUSH_STEPS_ENV,
            0,
        )
        self._decode_log_flush_seconds = env_non_negative_float(
            DECODE_LOG_FLUSH_SECONDS_ENV,
            0.0,
        )
        self._decode_log_last_flush = perf_counter()
        self._decode_step_index = 0
        self._pending_decode_steps: dict[int, _PendingDecodeStep] = {}
        self._completed_decode_steps: list[_CompletedDecodeStep] = []

    def _kv_block_usage(self) -> tuple[int, int]:
        pool = self.kv_cache_manager.block_pool
        # Block zero is vLLM's permanently allocated null block. Excluding it
        # matches BlockPool.get_usage() and reports usable logical KV blocks.
        total = max(0, int(pool.num_gpu_blocks) - 1)
        free = int(pool.get_num_free_blocks())
        return max(0, total - free), total

    def schedule(self) -> Any:
        if not self._decode_log_enabled:
            return super().schedule()

        step_start = perf_counter()
        scheduler_output = super().schedule()
        decode_info = single_token_decode_info(scheduler_output)
        if decode_info is not None:
            req_ids, seq_lens = decode_info
            kv_used, kv_total = self._kv_block_usage()
            self._pending_decode_steps[id(scheduler_output)] = _PendingDecodeStep(
                start_time=step_start,
                batch_size=len(req_ids),
                num_tokens=int(
                    scheduler_output.total_num_scheduled_tokens
                ),
                seq_lens=tuple(seq_lens),
                kv_used_blocks=kv_used,
                kv_total_blocks=kv_total,
            )
        return scheduler_output

    def update_from_output(
        self,
        scheduler_output: Any,
        model_runner_output: Any,
    ) -> Any:
        pending = (
            self._pending_decode_steps.pop(id(scheduler_output), None)
            if self._decode_log_enabled
            else None
        )
        outputs = super().update_from_output(
            scheduler_output,
            model_runner_output,
        )
        if pending is not None:
            self._decode_step_index += 1
            self._completed_decode_steps.append(
                _CompletedDecodeStep(
                    step=self._decode_step_index,
                    batch_size=pending.batch_size,
                    num_tokens=pending.num_tokens,
                    tpot_ms=(perf_counter() - pending.start_time) * 1000.0,
                    seq_lens=pending.seq_lens,
                    kv_used_blocks=pending.kv_used_blocks,
                    kv_total_blocks=pending.kv_total_blocks,
                )
            )

        # Batch stdout until idle or an explicitly configured periodic limit.
        # Per-step prints would enlarge gaps between decode steps in the trace.
        if self._decode_log_enabled and self._completed_decode_steps:
            flush_reason = self._decode_log_flush_reason()
            if flush_reason is not None:
                self._flush_decode_step_log(flush_reason)
        return outputs

    def _decode_log_flush_reason(self) -> str | None:
        if not self.has_unfinished_requests():
            return "idle"
        if (
            self._decode_log_flush_steps
            and len(self._completed_decode_steps)
            >= self._decode_log_flush_steps
        ):
            return "step_limit"
        if (
            self._decode_log_flush_seconds
            and perf_counter() - self._decode_log_last_flush
            >= self._decode_log_flush_seconds
        ):
            return "time_limit"
        return None

    def _flush_decode_step_log(self, reason: str) -> None:
        if not self._completed_decode_steps:
            return

        lines = [
            "VLLM_DECODE_STEP_LOG_BEGIN "
            f"dp_rank={self._decode_log_dp_rank} "
            f"count={len(self._completed_decode_steps)} reason={reason}"
        ]
        for record in self._completed_decode_steps:
            kv_percent = (
                100.0 * record.kv_used_blocks / record.kv_total_blocks
                if record.kv_total_blocks
                else 0.0
            )
            lines.append(
                f"[VLLM_DECODE step={record.step:04d} "
                f"dp_rank={self._decode_log_dp_rank}] "
                f"bsz={record.batch_size}, "
                f"num_tokens={record.num_tokens}, "
                f"TPOT={record.tpot_ms:.3f} ms, "
                f"seq_lens={list(record.seq_lens)}, "
                f"HBM_KV={record.kv_used_blocks}/"
                f"{record.kv_total_blocks} blocks ({kv_percent:.2f}%)"
            )
        lines.append("VLLM_DECODE_STEP_LOG_END")
        print("\n".join(lines), flush=True)
        self._completed_decode_steps.clear()
        self._decode_log_last_flush = perf_counter()

    def shutdown(self) -> None:
        if self._decode_log_enabled:
            self._flush_decode_step_log(reason="shutdown")
        super().shutdown()
