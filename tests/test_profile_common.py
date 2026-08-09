from __future__ import annotations

import unittest
from types import SimpleNamespace

from vllm_decode_profile.profile_common import (
    accepted_draft_token_counts,
    decode_step_info,
    format_decode_step_line,
    generated_token_counts,
    single_token_decode_info,
)


def make_scheduler_output(
    *,
    batch_size: int = 2,
    num_speculative_tokens: int = 0,
    include_spec_tokens: bool = True,
):
    req_ids = [f"req-{index}" for index in range(batch_size)]
    tokens_per_request = 1 + num_speculative_tokens
    scheduled = {req_id: tokens_per_request for req_id in req_ids}
    spec_tokens = (
        {
            req_id: list(range(num_speculative_tokens))
            for req_id in req_ids
        }
        if num_speculative_tokens and include_spec_tokens
        else {}
    )
    cached = SimpleNamespace(
        req_ids=req_ids,
        num_output_tokens=[1] * batch_size,
        num_computed_tokens=[100 + index for index in range(batch_size)],
    )
    return SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        scheduled_spec_decode_tokens=spec_tokens,
        scheduled_encoder_inputs={},
        num_scheduled_tokens=scheduled,
        total_num_scheduled_tokens=batch_size * tokens_per_request,
    )


class DecodeStepInfoTests(unittest.TestCase):
    def test_autoregressive_decode(self) -> None:
        output = make_scheduler_output()

        info = decode_step_info(output)

        self.assertIsNotNone(info)
        assert info is not None
        self.assertEqual(info.req_ids, ("req-0", "req-1"))
        self.assertEqual(info.seq_lens, (101, 102))
        self.assertEqual(info.total_num_scheduled_tokens, 2)
        self.assertEqual(
            single_token_decode_info(output),
            (["req-0", "req-1"], [101, 102]),
        )

    def test_mtp3_decode(self) -> None:
        output = make_scheduler_output(num_speculative_tokens=3)

        info = decode_step_info(output, num_speculative_tokens=3)

        self.assertIsNotNone(info)
        assert info is not None
        self.assertEqual(info.seq_lens, (104, 105))
        self.assertEqual(info.total_num_scheduled_tokens, 8)
        self.assertEqual(info.num_speculative_tokens, 3)

    def test_mtp3_rejects_bootstrap_decode_without_drafts(self) -> None:
        output = make_scheduler_output(
            num_speculative_tokens=3,
            include_spec_tokens=False,
        )

        self.assertIsNone(
            decode_step_info(output, num_speculative_tokens=3)
        )

    def test_mtp3_rejects_wrong_draft_width(self) -> None:
        output = make_scheduler_output(num_speculative_tokens=3)
        output.scheduled_spec_decode_tokens["req-0"] = [1, 2]

        self.assertIsNone(
            decode_step_info(output, num_speculative_tokens=3)
        )

    def test_expected_batch_size_is_enforced(self) -> None:
        output = make_scheduler_output(num_speculative_tokens=3)

        self.assertIsNone(
            decode_step_info(
                output,
                expected_batch_size=3,
                num_speculative_tokens=3,
            )
        )

    def test_autoregressive_mode_rejects_speculative_step(self) -> None:
        output = make_scheduler_output(num_speculative_tokens=3)

        self.assertIsNone(decode_step_info(output))

    def test_negative_speculative_width_is_invalid(self) -> None:
        output = make_scheduler_output()

        with self.assertRaises(ValueError):
            decode_step_info(output, num_speculative_tokens=-1)


class GeneratedTokenCountTests(unittest.TestCase):
    def test_counts_are_returned_in_requested_order(self) -> None:
        output = SimpleNamespace(
            req_id_to_index={"req-a": 1, "req-b": 0},
            sampled_token_ids=[[10, 11], [20, 21, 22, 23]],
        )

        counts = generated_token_counts(output, ("req-a", "req-b"))

        self.assertEqual(counts, (4, 2))

    def test_missing_request_has_zero_count(self) -> None:
        output = SimpleNamespace(
            req_id_to_index={"req-a": 0},
            sampled_token_ids=[[10]],
        )

        counts = generated_token_counts(output, ("req-a", "missing"))

        self.assertEqual(counts, (1, 0))

    def test_accepted_drafts_exclude_the_non_draft_output(self) -> None:
        accepted = accepted_draft_token_counts((4, 3, 2, 1), 3)

        self.assertEqual(accepted, (3, 2, 1, 0))

    def test_accepted_drafts_are_clamped(self) -> None:
        accepted = accepted_draft_token_counts((10, 0), 3)

        self.assertEqual(accepted, (3, 0))

    def test_negative_draft_width_is_invalid(self) -> None:
        with self.assertRaises(ValueError):
            accepted_draft_token_counts((1,), -1)


class DecodeStepFormattingTests(unittest.TestCase):
    def test_autoregressive_format_is_unchanged(self) -> None:
        line = format_decode_step_line(
            step=1,
            batch_size=2,
            num_tokens=2,
            latency_ms=68.42,
            seq_lens=(101, 102),
            kv_used_blocks=75,
            kv_total_blocks=100,
        )

        self.assertEqual(
            line,
            "[VLLM_DECODE step=0001] bsz=2, num_tokens=2, "
            "TPOT=68.420 ms, seq_lens=[101, 102], "
            "HBM_KV=75/100 blocks (75.00%)",
        )

    def test_mtp3_format_uses_step_and_acceptance(self) -> None:
        line = format_decode_step_line(
            step=1,
            batch_size=6,
            num_tokens=6,
            latency_ms=80.0,
            seq_lens=(101, 102, 103, 104, 105, 106),
            kv_used_blocks=1407,
            kv_total_blocks=1680,
            num_speculative_tokens=3,
            accepted_draft_tokens=10,
        )

        self.assertEqual(
            line,
            "[VLLM_DECODE step=0001] bsz=6, num_tokens=6, "
            "STEP=80.000 ms, accept/draft=10/18, "
            "seq_lens=[101, 102, 103, 104, 105, 106], "
            "HBM_KV=1407/1680 blocks (83.75%)",
        )


if __name__ == "__main__":
    unittest.main()
