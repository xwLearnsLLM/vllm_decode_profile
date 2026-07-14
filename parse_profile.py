"""Parse the single rank-0 Ascend profile under VLLM_PROFILE_DIR."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


RAW_PROFILE_SUFFIX = "ascend_pt"
ANALYSIS_DIR_NAME = "ASCEND_PROFILER_OUTPUT"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find and parse the single rank-0 *ascend_pt directory produced "
            "by profile_vllm.py."
        )
    )
    parser.add_argument(
        "profile_dir",
        nargs="?",
        help=(
            "Profile root directory. Defaults to the VLLM_PROFILE_DIR "
            "environment variable."
        ),
    )
    return parser.parse_args()


def _resolve_profile_root(value: str | None) -> Path:
    value = value or os.environ.get("VLLM_PROFILE_DIR")
    if not value:
        raise SystemExit(
            "VLLM_PROFILE_DIR is not set. Export the same directory used by "
            "profile_vllm.py, then rerun this script."
        )

    profile_root = Path(value).expanduser().resolve()
    if not profile_root.is_dir():
        raise SystemExit(f"Profile directory does not exist: {profile_root}")
    return profile_root


def _find_unique_raw_profile(profile_root: Path) -> Path:
    candidates = []
    if profile_root.name.endswith(RAW_PROFILE_SUFFIX):
        candidates.append(profile_root)
    candidates.extend(
        path
        for path in profile_root.rglob(f"*{RAW_PROFILE_SUFFIX}")
        if path.is_dir()
    )
    candidates = sorted(set(candidates), key=lambda path: str(path))

    if not candidates:
        raise SystemExit(
            f"No *{RAW_PROFILE_SUFFIX} directory found under: "
            f"{profile_root}"
        )
    if len(candidates) != 1:
        formatted = "\n".join(f"  - {path}" for path in candidates)
        raise SystemExit(
            "Expected exactly one rank-0 raw profile directory, found "
            f"{len(candidates)} under {profile_root}:\n{formatted}"
        )
    return candidates[0]


def _find_analysis_dirs(raw_profile_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in raw_profile_dir.rglob(ANALYSIS_DIR_NAME)
            if path.is_dir()
        ),
        key=lambda path: str(path),
    )


def main() -> None:
    args = _parse_args()
    profile_root = _resolve_profile_root(args.profile_dir)
    raw_profile_dir = _find_unique_raw_profile(profile_root)

    print(f"profile root: {profile_root}")
    print(f"raw profile : {raw_profile_dir}")

    analysis_dirs = _find_analysis_dirs(raw_profile_dir)
    if analysis_dirs:
        print("profile is already parsed; skipping analyse()")
    else:
        try:
            from torch_npu.profiler.profiler import analyse
        except ImportError as exc:
            raise SystemExit(
                "Unable to import torch_npu.profiler.profiler.analyse. Run "
                "this script in the same Ascend Python environment used for "
                "inference."
            ) from exc

        print("parsing profile; this may take a while...")
        analyse(str(raw_profile_dir))
        analysis_dirs = _find_analysis_dirs(raw_profile_dir)
        if not analysis_dirs:
            raise SystemExit(
                "analyse() returned, but no ASCEND_PROFILER_OUTPUT directory "
                f"was created under: {raw_profile_dir}"
            )

    for analysis_dir in analysis_dirs:
        print(f"analysis output: {analysis_dir}")
        trace_files = sorted(analysis_dir.rglob("trace_view.json"))
        if trace_files:
            for trace_file in trace_files:
                print(f"trace view     : {trace_file}")
        else:
            print("trace view     : not found")


if __name__ == "__main__":
    main()
