"""Pure helpers shared by the profiling worker and scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DecodeStepInfo:
    """Shape information for one uniform autoregressive or MTP step."""

    req_ids: tuple[str, ...]
    seq_lens: tuple[int, ...]
    total_num_scheduled_tokens: int
    num_speculative_tokens: int


def decode_step_info(
    scheduler_output: Any,
    expected_batch_size: int | None = None,
    num_speculative_tokens: int = 0,
) -> DecodeStepInfo | None:
    """Return information for a pure, uniform decode verification step.

    With ``num_speculative_tokens == 0``, every request must schedule one
    ordinary decode token. With MTP enabled, every request must schedule one
    ordinary token plus exactly ``num_speculative_tokens`` draft tokens.

    Each returned sequence length is ``num_computed_tokens +
    num_scheduled_tokens``. For MTP this is the end of the target-model
    verification span, before rejected draft tokens are rolled back.
    """

    if num_speculative_tokens < 0:
        raise ValueError("num_speculative_tokens must be non-negative")

    if scheduler_output.scheduled_new_reqs:
        return None
    if getattr(scheduler_output, "scheduled_encoder_inputs", None):
        return None

    cached = scheduler_output.scheduled_cached_reqs
    req_ids = tuple(cached.req_ids)
    if not req_ids:
        return None
    if expected_batch_size is not None and len(req_ids) != expected_batch_size:
        return None
    if len(cached.num_output_tokens) != len(req_ids):
        return None
    if len(cached.num_computed_tokens) != len(req_ids):
        return None
    if any(int(count) <= 0 for count in cached.num_output_tokens):
        return None

    scheduled = scheduler_output.num_scheduled_tokens
    scheduled_spec_tokens = (
        getattr(scheduler_output, "scheduled_spec_decode_tokens", None) or {}
    )
    tokens_per_request = 1 + num_speculative_tokens

    if num_speculative_tokens:
        if set(scheduled_spec_tokens) != set(req_ids):
            return None
        if any(
            len(scheduled_spec_tokens[req_id]) != num_speculative_tokens
            for req_id in req_ids
        ):
            return None
    elif scheduled_spec_tokens:
        return None

    if any(
        int(scheduled.get(req_id, 0)) != tokens_per_request
        for req_id in req_ids
    ):
        return None

    expected_total = len(req_ids) * tokens_per_request
    if int(scheduler_output.total_num_scheduled_tokens) != expected_total:
        return None

    seq_lens = tuple(
        int(num_computed) + int(scheduled[req_id])
        for req_id, num_computed in zip(
            req_ids,
            cached.num_computed_tokens,
        )
    )
    return DecodeStepInfo(
        req_ids=req_ids,
        seq_lens=seq_lens,
        total_num_scheduled_tokens=expected_total,
        num_speculative_tokens=num_speculative_tokens,
    )


def single_token_decode_info(
    scheduler_output: Any,
    expected_batch_size: int | None = None,
) -> tuple[list[str], list[int]] | None:
    """Return request IDs and current forward lengths for a pure decode step.

    The returned length is ``num_computed_tokens + num_scheduled_tokens``:
    it includes the token processed by this forward, but not the token sampled
    at the end of the forward.
    """

    info = decode_step_info(
        scheduler_output,
        expected_batch_size=expected_batch_size,
        num_speculative_tokens=0,
    )
    if info is None:
        return None
    return list(info.req_ids), list(info.seq_lens)


def generated_token_counts(
    model_runner_output: Any,
    req_ids: tuple[str, ...],
) -> tuple[int, ...]:
    """Return target output lengths in request order for one engine step."""

    sampled_token_ids = getattr(model_runner_output, "sampled_token_ids", None)
    req_id_to_index = getattr(model_runner_output, "req_id_to_index", None)
    if not sampled_token_ids or not req_id_to_index:
        return tuple(0 for _ in req_ids)

    counts = []
    for req_id in req_ids:
        index = req_id_to_index.get(req_id)
        if index is None or not 0 <= int(index) < len(sampled_token_ids):
            counts.append(0)
            continue
        token_ids = sampled_token_ids[int(index)]
        counts.append(len(token_ids) if token_ids is not None else 0)
    return tuple(counts)


def accepted_draft_token_counts(
    generated_counts: tuple[int, ...],
    num_speculative_tokens: int,
) -> tuple[int, ...]:
    """Return accepted draft-token counts for each request.

    vLLM's speculative output contains one non-draft output token in addition
    to the accepted draft tokens. Clamp the result defensively to the number
    of drafts that were scheduled.
    """

    if num_speculative_tokens < 0:
        raise ValueError("num_speculative_tokens must be non-negative")
    return tuple(
        min(num_speculative_tokens, max(0, count - 1))
        for count in generated_counts
    )


def format_decode_step_line(
    *,
    step: int,
    batch_size: int,
    num_tokens: int,
    latency_ms: float,
    seq_lens: tuple[int, ...],
    kv_used_blocks: int,
    kv_total_blocks: int,
    num_speculative_tokens: int = 0,
    accepted_draft_tokens: int = 0,
) -> str:
    """Format one ordinary or MTP decode measurement."""

    kv_percent = (
        100.0 * kv_used_blocks / kv_total_blocks
        if kv_total_blocks
        else 0.0
    )
    prefix = (
        f"[VLLM_DECODE step={step:04d}] "
        f"bsz={batch_size}, num_tokens={num_tokens}, "
    )
    if num_speculative_tokens:
        total_draft_tokens = batch_size * num_speculative_tokens
        latency = (
            f"STEP={latency_ms:.3f} ms, "
            f"accept/draft={accepted_draft_tokens}/{total_draft_tokens}, "
        )
    else:
        latency = f"TPOT={latency_ms:.3f} ms, "
    return (
        f"{prefix}{latency}seq_lens={list(seq_lens)}, "
        f"HBM_KV={kv_used_blocks}/{kv_total_blocks} blocks "
        f"({kv_percent:.2f}%)"
    )
