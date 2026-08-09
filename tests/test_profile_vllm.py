from __future__ import annotations

import io
import os
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from vllm_decode_profile.profile_vllm import (
    BASELINE_ALL2ALL_BACKEND,
    SCHEDULER_CLASS,
    _max_gen_tokens,
    _print_effective_runtime_config,
    env_json_object,
)


def make_fake_llm(*, capture_size: int, speculative_config=None):
    vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode="CUDAGraphMode.FULL_DECODE_ONLY",
            cudagraph_capture_sizes=[capture_size],
        ),
        parallel_config=SimpleNamespace(
            all2all_backend=BASELINE_ALL2ALL_BACKEND,
        ),
        scheduler_config=SimpleNamespace(scheduler_cls=SCHEDULER_CLASS),
        speculative_config=speculative_config,
    )
    return SimpleNamespace(
        llm_engine=SimpleNamespace(vllm_config=vllm_config),
    )


class LauncherConfigurationTests(unittest.TestCase):
    def test_mode_specific_generation_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_max_gen_tokens(False), 8)
            self.assertEqual(_max_gen_tokens(True), 16)

    def test_json_object_environment_value(self) -> None:
        with patch.dict(
            os.environ,
            {"TEST_JSON_CONFIG": '{"enabled": true}'},
            clear=True,
        ):
            self.assertEqual(
                env_json_object("TEST_JSON_CONFIG"),
                {"enabled": True},
            )

    def test_json_array_is_rejected(self) -> None:
        with patch.dict(
            os.environ,
            {"TEST_JSON_CONFIG": "[]"},
            clear=True,
        ):
            with self.assertRaises(ValueError):
                env_json_object("TEST_JSON_CONFIG")

    def test_mtp3_runtime_config_is_accepted(self) -> None:
        speculative_config = SimpleNamespace(
            method="mtp",
            num_speculative_tokens=3,
            enforce_eager=True,
        )
        llm = make_fake_llm(
            capture_size=24,
            speculative_config=speculative_config,
        )

        with redirect_stdout(io.StringIO()):
            _print_effective_runtime_config(
                llm,
                enable_expert_parallel=True,
                batch_size=6,
                enable_mtp=True,
                mtp_drafter_enforce_eager=True,
            )

    def test_mtp3_requires_four_tokens_per_request_capture(self) -> None:
        speculative_config = SimpleNamespace(
            method="mtp",
            num_speculative_tokens=3,
            enforce_eager=True,
        )
        llm = make_fake_llm(
            capture_size=6,
            speculative_config=speculative_config,
        )

        with redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):
                _print_effective_runtime_config(
                    llm,
                    enable_expert_parallel=True,
                    batch_size=6,
                    enable_mtp=True,
                    mtp_drafter_enforce_eager=True,
                )


if __name__ == "__main__":
    unittest.main()
