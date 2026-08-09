"""EngineCore decode-step statistics for the vLLM-Ascend baseline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from vllm.v1.core.sched.scheduler import Scheduler

from vllm_decode_profile.profile_common import (
    accepted_draft_token_counts,
    decode_step_info,
    format_decode_step_line,
    generated_token_counts,
)


SPECULATIVE_TOKENS_ENV = "VLLM_PROFILE_NUM_SPECULATIVE_TOKENS"


def _profile_num_speculative_tokens() -> int:
    value = os.environ.get(SPECULATIVE_TOKENS_ENV, "0")
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(
            f"{SPECULATIVE_TOKENS_ENV} must be an integer, got {value!r}"
        ) from exc
    if parsed not in (0, 3):
        raise RuntimeError(
            f"{SPECULATIVE_TOKENS_ENV} supports only 0 or 3, got {parsed}"
        )
    return parsed


@dataclass(frozen=True)
class _PendingDecodeStep:
    iteration_start_time: float
    model_start_time: float
    req_ids: tuple[str, ...]
    batch_size: int
    num_tokens: int
    num_speculative_tokens: int
    seq_lens: tuple[int, ...]
    kv_used_blocks: int
    kv_total_blocks: int


@dataclass(frozen=True)
class _CompletedDecodeStep:
    step: int
    batch_size: int
    num_tokens: int
    num_speculative_tokens: int
    latency_ms: float
    accepted_draft_tokens: int
    seq_lens: tuple[int, ...]
    kv_used_blocks: int
    kv_total_blocks: int


class DecodeStepLoggingScheduler(Scheduler):
    """Measure pure decode EngineCore steps without per-step stdout I/O."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._num_speculative_tokens = _profile_num_speculative_tokens()
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
        iteration_start = perf_counter()
        scheduler_output = super().schedule()
        decode_info = decode_step_info(
            scheduler_output,
            num_speculative_tokens=self._num_speculative_tokens,
        )
        if decode_info is not None:
            kv_used, kv_total = self._kv_block_usage()
            # This is the closest scheduler-side boundary to the following
            # model_executor.execute_model() call. In update_from_output() we
            # stop it as soon as the model result reaches the scheduler.
            model_start = perf_counter()
            self._pending_decode_steps[id(scheduler_output)] = _PendingDecodeStep(
                iteration_start_time=iteration_start,
                model_start_time=model_start,
                req_ids=decode_info.req_ids,
                batch_size=len(decode_info.req_ids),
                # Keep the historical meaning: ordinary tokens in this step.
                # MTP draft tokens are reported separately as accept/draft.
                num_tokens=len(decode_info.req_ids),
                num_speculative_tokens=decode_info.num_speculative_tokens,
                seq_lens=decode_info.seq_lens,
                kv_used_blocks=kv_used,
                kv_total_blocks=kv_total,
            )
        return scheduler_output

    def update_from_output(
        self,
        scheduler_output: Any,
        model_runner_output: Any,
    ) -> Any:
        pending = self._pending_decode_steps.pop(
            id(scheduler_output),
            None,
        )
        model_end = perf_counter()
        accepted_token_lens = (
            generated_token_counts(model_runner_output, pending.req_ids)
            if pending is not None
            else ()
        )
        if pending is not None and pending.num_speculative_tokens:
            maximum_output_tokens = 1 + pending.num_speculative_tokens
            if any(
                count < 1 or count > maximum_output_tokens
                for count in accepted_token_lens
            ):
                raise RuntimeError(
                    "Invalid MTP model output lengths for acceptance "
                    f"measurement: {accepted_token_lens!r}"
                )
        outputs = super().update_from_output(
            scheduler_output,
            model_runner_output,
        )
        if pending is not None:
            accepted_drafts_per_request = accepted_draft_token_counts(
                accepted_token_lens,
                pending.num_speculative_tokens,
            )
            accepted_draft_tokens = sum(accepted_drafts_per_request)
            if pending.num_speculative_tokens:
                latency_ms = (
                    model_end - pending.model_start_time
                ) * 1000.0
                # The verification span includes all drafts. Report the
                # effective sequence lengths after rejected drafts roll back.
                seq_lens = tuple(
                    seq_len
                    - (
                        pending.num_speculative_tokens
                        - accepted_drafts
                    )
                    for seq_len, accepted_drafts in zip(
                        pending.seq_lens,
                        accepted_drafts_per_request,
                    )
                )
            else:
                # Preserve the original non-MTP TPOT boundary, including
                # scheduler work before and after model execution.
                latency_ms = (
                    perf_counter() - pending.iteration_start_time
                ) * 1000.0
                seq_lens = pending.seq_lens
            self._decode_step_index += 1
            self._completed_decode_steps.append(
                _CompletedDecodeStep(
                    step=self._decode_step_index,
                    batch_size=pending.batch_size,
                    num_tokens=pending.num_tokens,
                    num_speculative_tokens=pending.num_speculative_tokens,
                    latency_ms=latency_ms,
                    accepted_draft_tokens=accepted_draft_tokens,
                    seq_lens=seq_lens,
                    kv_used_blocks=pending.kv_used_blocks,
                    kv_total_blocks=pending.kv_total_blocks,
                )
            )

        # Defer all stdout until the request set is finished. Immediate prints
        # would enlarge the idle gap before the next decode step in the trace.
        if not self.has_unfinished_requests():
            self._flush_decode_step_log()
        return outputs

    def _flush_decode_step_log(self) -> None:
        if not self._completed_decode_steps:
            return

        lines = [
            "VLLM_DECODE_STEP_LOG_BEGIN "
            f"count={len(self._completed_decode_steps)} "
            f"mode={'mtp3' if self._num_speculative_tokens else 'decode'}"
        ]
        for record in self._completed_decode_steps:
            lines.append(
                format_decode_step_line(
                    step=record.step,
                    batch_size=record.batch_size,
                    num_tokens=record.num_tokens,
                    latency_ms=record.latency_ms,
                    seq_lens=record.seq_lens,
                    kv_used_blocks=record.kv_used_blocks,
                    kv_total_blocks=record.kv_total_blocks,
                    num_speculative_tokens=record.num_speculative_tokens,
                    accepted_draft_tokens=record.accepted_draft_tokens,
                )
            )
        lines.append("VLLM_DECODE_STEP_LOG_END")
        print("\n".join(lines), flush=True)
        self._completed_decode_steps.clear()

    def shutdown(self) -> None:
        self._flush_decode_step_log()
        super().shutdown()
