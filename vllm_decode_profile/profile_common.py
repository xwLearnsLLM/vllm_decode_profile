"""Pure helpers shared by the profiling worker and scheduler."""

from __future__ import annotations

from typing import Any


def single_token_decode_info(
    scheduler_output: Any,
    expected_batch_size: int | None = None,
) -> tuple[list[str], list[int]] | None:
    """Return request IDs and current forward lengths for a pure decode step.

    The returned length is ``num_computed_tokens + num_scheduled_tokens``:
    it includes the token processed by this forward, but not the token sampled
    at the end of the forward.
    """

    if scheduler_output.scheduled_new_reqs:
        return None
    if getattr(scheduler_output, "scheduled_spec_decode_tokens", None):
        return None
    if getattr(scheduler_output, "scheduled_encoder_inputs", None):
        return None

    cached = scheduler_output.scheduled_cached_reqs
    req_ids = list(cached.req_ids)
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
    if int(scheduler_output.total_num_scheduled_tokens) != len(req_ids):
        return None

    scheduled = scheduler_output.num_scheduled_tokens
    if any(int(scheduled.get(req_id, 0)) != 1 for req_id in req_ids):
        return None

    seq_lens = [
        int(num_computed) + int(scheduled[req_id])
        for req_id, num_computed in zip(
            req_ids,
            cached.num_computed_tokens,
        )
    ]
    return req_ids, seq_lens
