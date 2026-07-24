from __future__ import annotations

import importlib
import io
import os
import sys
import types
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch


class _FakeNPUWorker:
    def __init__(
        self,
        *,
        dp_rank: int,
        dp_size: int = 32,
        tp_size: int = 1,
        rank: int = 0,
    ) -> None:
        self.parallel_config = SimpleNamespace(
            data_parallel_rank=dp_rank,
            data_parallel_size=dp_size,
            tensor_parallel_size=tp_size,
        )
        self.rank = rank
        self.events: list[tuple] = []

    def profile(
        self,
        is_start: bool = True,
        profile_prefix: str | None = None,
    ) -> None:
        self.events.append(("profile", is_start, profile_prefix))

    def execute_model(self, scheduler_output):
        self.events.append(("execute", scheduler_output))
        return "model-output"


class _FakeBlockPool:
    num_gpu_blocks = 101

    def get_num_free_blocks(self) -> int:
        return 40


class _FakeScheduler:
    def __init__(
        self,
        *,
        dp_rank: int,
        dp_size: int = 32,
        scheduler_outputs: list | None = None,
    ) -> None:
        self.parallel_config = SimpleNamespace(
            data_parallel_rank=dp_rank,
            data_parallel_size=dp_size,
        )
        self.kv_cache_manager = SimpleNamespace(
            block_pool=_FakeBlockPool()
        )
        self._scheduler_outputs = list(scheduler_outputs or [])
        self._unfinished = False
        self.shutdown_called = False

    def schedule(self):
        return self._scheduler_outputs.pop(0)

    def update_from_output(self, scheduler_output, model_runner_output):
        self._unfinished = bool(scheduler_output.unfinished_after)
        return "updated"

    def has_unfinished_requests(self) -> bool:
        return self._unfinished

    def shutdown(self) -> None:
        self.shutdown_called = True


def _install_fake_dependency_modules() -> None:
    modules = {
        "vllm_ascend": types.ModuleType("vllm_ascend"),
        "vllm_ascend.worker": types.ModuleType("vllm_ascend.worker"),
        "vllm_ascend.worker.worker": types.ModuleType(
            "vllm_ascend.worker.worker"
        ),
        "vllm": types.ModuleType("vllm"),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.core": types.ModuleType("vllm.v1.core"),
        "vllm.v1.core.sched": types.ModuleType("vllm.v1.core.sched"),
        "vllm.v1.core.sched.scheduler": types.ModuleType(
            "vllm.v1.core.sched.scheduler"
        ),
    }
    modules["vllm_ascend.worker.worker"].NPUWorker = _FakeNPUWorker
    modules["vllm.v1.core.sched.scheduler"].Scheduler = _FakeScheduler
    sys.modules.update(modules)


def _scheduler_output(
    batch_size: int,
    *,
    has_new_request: bool = False,
    unfinished_after: bool = False,
):
    req_ids = [f"req-{index}" for index in range(batch_size)]
    return SimpleNamespace(
        scheduled_new_reqs=["new"] if has_new_request else [],
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=req_ids,
            num_output_tokens=[1] * batch_size,
            num_computed_tokens=[100 + index for index in range(batch_size)],
        ),
        total_num_scheduled_tokens=batch_size,
        num_scheduled_tokens={req_id: 1 for req_id in req_ids},
        unfinished_after=unfinished_after,
    )


class ProfileWorkerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _install_fake_dependency_modules()
        sys.modules.pop("vllm_decode_profile.profile_worker", None)
        cls.module = importlib.import_module(
            "vllm_decode_profile.profile_worker"
        )

    def test_only_selected_dp_rank_profiles_and_auto_stops(self) -> None:
        env = {
            "VLLM_PROFILE_TARGET_DP_RANK": "0",
            "VLLM_PROFILE_TARGET_TP_RANK": "0",
            "VLLM_PROFILE_TRIGGER_BATCH_SIZE": "2",
            "VLLM_PROFILE_MAX_DECODE_STEPS": "2",
            "VLLM_PROFILE_STOP_ON_PHASE_CHANGE": "1",
        }
        with patch.dict(os.environ, env, clear=True):
            target = self.module.DecodeOnlyRankFilteredNPUWorker(dp_rank=0)
            other = self.module.DecodeOnlyRankFilteredNPUWorker(dp_rank=1)

            target.profile(is_start=True)
            other.profile(is_start=True)

            decode = _scheduler_output(2)
            target.execute_model(decode)
            target.execute_model(decode)
            other.execute_model(decode)

            self.assertEqual(
                [event[0:2] for event in target.events],
                [
                    ("profile", True),
                    ("execute", decode),
                    ("execute", decode),
                    ("profile", False),
                ],
            )
            self.assertEqual(other.events, [("execute", decode)])

            # The public stop endpoint may arrive after the automatic stop.
            target.profile(is_start=False)
            self.assertEqual(len(target.events), 4)

    def test_phase_change_stops_before_non_decode_execution(self) -> None:
        env = {
            "VLLM_PROFILE_TARGET_DP_RANK": "0",
            "VLLM_PROFILE_TARGET_TP_RANK": "0",
            "VLLM_PROFILE_TRIGGER_BATCH_SIZE": "2",
            "VLLM_PROFILE_MAX_DECODE_STEPS": "0",
            "VLLM_PROFILE_STOP_ON_PHASE_CHANGE": "1",
        }
        with patch.dict(os.environ, env, clear=True):
            worker = self.module.DecodeOnlyRankFilteredNPUWorker(dp_rank=0)
            decode = _scheduler_output(2)
            mixed = _scheduler_output(2, has_new_request=True)

            worker.profile(is_start=True)
            worker.execute_model(decode)
            worker.execute_model(mixed)

            self.assertEqual(
                [event[0:2] for event in worker.events],
                [
                    ("profile", True),
                    ("execute", decode),
                    ("profile", False),
                    ("execute", mixed),
                ],
            )

    def test_legacy_offline_tp_rank_remains_supported(self) -> None:
        env = {
            "VLLM_PROFILE_GLOBAL_RANK": "1",
            "VLLM_PROFILE_EXPECTED_BATCH_SIZE": "2",
        }
        with patch.dict(os.environ, env, clear=True):
            worker = self.module.DecodeOnlyRankFilteredNPUWorker(
                dp_rank=0,
                dp_size=1,
                tp_size=2,
                rank=1,
            )
            decode = _scheduler_output(2)

            worker.profile(is_start=True, profile_prefix="offline")
            worker.execute_model(decode)
            worker.profile(is_start=False)

            self.assertEqual(
                [event[0:2] for event in worker.events],
                [
                    ("profile", True),
                    ("execute", decode),
                    ("profile", False),
                ],
            )


class ProfileSchedulerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _install_fake_dependency_modules()
        sys.modules.pop("vllm_decode_profile.profile_scheduler", None)
        cls.module = importlib.import_module(
            "vllm_decode_profile.profile_scheduler"
        )

    def test_only_selected_dp_scheduler_logs(self) -> None:
        env = {
            "VLLM_PROFILE_TARGET_DP_RANK": "0",
            "VLLM_DECODE_LOG_DP_RANK": "0",
            "VLLM_DECODE_LOG_FLUSH_STEPS": "0",
            "VLLM_DECODE_LOG_FLUSH_SECONDS": "0",
        }
        with patch.dict(os.environ, env, clear=True):
            target_output = _scheduler_output(2, unfinished_after=False)
            other_output = _scheduler_output(2, unfinished_after=False)
            target = self.module.DecodeStepLoggingScheduler(
                dp_rank=0,
                scheduler_outputs=[target_output],
            )
            other = self.module.DecodeStepLoggingScheduler(
                dp_rank=1,
                scheduler_outputs=[other_output],
            )

            stream = io.StringIO()
            with redirect_stdout(stream):
                scheduled = target.schedule()
                target.update_from_output(scheduled, object())
                scheduled = other.schedule()
                other.update_from_output(scheduled, object())

            log = stream.getvalue()
            self.assertIn("dp_rank=0", log)
            self.assertIn("bsz=2", log)
            self.assertNotIn("dp_rank=1", log)
            self.assertEqual(other._completed_decode_steps, [])

    def test_continuous_load_can_flush_by_step_limit(self) -> None:
        env = {
            "VLLM_PROFILE_TARGET_DP_RANK": "0",
            "VLLM_DECODE_LOG_DP_RANK": "0",
            "VLLM_DECODE_LOG_FLUSH_STEPS": "1",
            "VLLM_DECODE_LOG_FLUSH_SECONDS": "0",
        }
        with patch.dict(os.environ, env, clear=True):
            output = _scheduler_output(1, unfinished_after=True)
            scheduler = self.module.DecodeStepLoggingScheduler(
                dp_rank=0,
                scheduler_outputs=[output],
            )

            stream = io.StringIO()
            with redirect_stdout(stream):
                scheduled = scheduler.schedule()
                scheduler.update_from_output(scheduled, object())

            self.assertIn("reason=step_limit", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
