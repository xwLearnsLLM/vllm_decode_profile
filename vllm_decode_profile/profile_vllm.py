"""Collect a rank-0, decode-only profile of the vLLM-Ascend baseline."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


DEFAULT_MODEL_PATH = "/home/models/DeepSeek-V3.2-REAP-345B-A37B-BF16/"
DEFAULT_TP_SIZE = 16
DEFAULT_PROMPT_LENGTHS = "8200,8201"
DEFAULT_MAX_GEN_TOKENS = 8
DEFAULT_MTP_MAX_GEN_TOKENS = 16
DEFAULT_MAX_NUM_BATCHED_TOKENS = 8192
MTP_SPECULATIVE_TOKENS = 3
BASELINE_ALL2ALL_BACKEND = "flashinfer_all2allv"
PROFILE_SPECULATIVE_TOKENS_ENV = (
    "VLLM_PROFILE_NUM_SPECULATIVE_TOKENS"
)
DEFAULT_GLM_MTP_ADDITIONAL_CONFIG = {
    "fuse_muls_add": True,
    "multistream_overlap_shared_expert": True,
    "ascend_compilation_config": {"enable_npugraph_ex": True},
}
WORKER_CLASS = (
    "vllm_decode_profile.profile_worker.DecodeOnlyRankFilteredNPUWorker"
)
SCHEDULER_CLASS = (
    "vllm_decode_profile.profile_scheduler.DecodeStepLoggingScheduler"
)
REPO_ROOT = Path(__file__).resolve().parents[1]


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None else float(value)


def env_json_object(
    name: str,
    default: dict | None = None,
) -> dict | None:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return parsed


def parse_prompt_lengths() -> list[int]:
    value = os.environ.get("VLLM_PROMPT_LENGTHS", DEFAULT_PROMPT_LENGTHS)
    lengths = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError(
            "VLLM_PROMPT_LENGTHS must contain positive integers"
        )
    return lengths


def _model_path() -> str:
    return os.environ.get("VLLM_MODEL", DEFAULT_MODEL_PATH)


def _tp_size() -> int:
    return env_int("VLLM_TP_SIZE", DEFAULT_TP_SIZE)


def _max_gen_tokens(enable_mtp: bool) -> int:
    default = (
        DEFAULT_MTP_MAX_GEN_TOKENS
        if enable_mtp
        else DEFAULT_MAX_GEN_TOKENS
    )
    return env_int("VLLM_MAX_GEN_TOKENS", default)


def _profile_dir(
    tp_size: int,
    batch_size: int,
    enable_mtp: bool,
) -> Path:
    decode_mode = "mtp3" if enable_mtp else "decode"
    default_dir = (
        REPO_ROOT
        / "profiles"
        / (
            f"vllm_baseline_tp{tp_size}_bs{batch_size}_{decode_mode}_"
            "full_decode_only_"
            f"{time.strftime('%Y%m%d_%H%M%S')}"
        )
    )
    return Path(
        os.environ.get("VLLM_PROFILE_DIR", str(default_dir))
    ).expanduser().resolve()


def _ensure_project_is_importable() -> None:
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    pythonpath = os.environ.get("PYTHONPATH", "")
    entries = [entry for entry in pythonpath.split(os.pathsep) if entry]
    if repo_root not in entries:
        os.environ["PYTHONPATH"] = os.pathsep.join([repo_root, *entries])


def _print_effective_runtime_config(
    llm,
    enable_expert_parallel: bool,
    batch_size: int,
    enable_mtp: bool,
    mtp_drafter_enforce_eager: bool,
) -> None:
    config = llm.llm_engine.vllm_config.compilation_config
    mode = getattr(config, "cudagraph_mode", None)
    capture_sizes = getattr(config, "cudagraph_capture_sizes", None)
    parallel_config = llm.llm_engine.vllm_config.parallel_config
    all2all_backend = getattr(parallel_config, "all2all_backend", None)
    scheduler_cls = getattr(
        llm.llm_engine.vllm_config.scheduler_config,
        "scheduler_cls",
        None,
    )
    speculative_config = llm.llm_engine.vllm_config.speculative_config
    speculative_method = getattr(speculative_config, "method", None)
    speculative_tokens = getattr(
        speculative_config,
        "num_speculative_tokens",
        0,
    )
    speculative_enforce_eager = getattr(
        speculative_config,
        "enforce_eager",
        None,
    )
    expected_capture_size = batch_size * (
        1 + (MTP_SPECULATIVE_TOKENS if enable_mtp else 0)
    )
    print(
        "effective runtime config: "
        f"cudagraph_mode={mode}, capture_sizes={capture_sizes}, "
        f"all2all_backend={all2all_backend}, scheduler_cls={scheduler_cls}"
        f", speculative_method={speculative_method}, "
        f"speculative_tokens={speculative_tokens}, "
        f"speculative_enforce_eager={speculative_enforce_eager}"
    )
    if "FULL_DECODE_ONLY" not in str(mode):
        raise RuntimeError(
            "vLLM did not retain cudagraph_mode=FULL_DECODE_ONLY; "
            "do not use this run as the graph baseline"
        )
    if capture_sizes is None or expected_capture_size not in {
        int(size) for size in capture_sizes
    }:
        raise RuntimeError(
            "vLLM did not retain the requested decode graph capture size "
            f"{expected_capture_size}; got {capture_sizes!r}"
        )
    if (
        enable_expert_parallel
        and all2all_backend != BASELINE_ALL2ALL_BACKEND
    ):
        raise RuntimeError(
            "vLLM did not retain the vLLM-Ascend baseline EP backend "
            f"{BASELINE_ALL2ALL_BACKEND!r}; got {all2all_backend!r}"
        )
    if SCHEDULER_CLASS not in str(scheduler_cls):
        raise RuntimeError(
            "vLLM did not retain the decode-step logging scheduler; "
            f"got {scheduler_cls!r}"
        )
    if enable_mtp:
        if speculative_method != "mtp":
            raise RuntimeError(
                "vLLM did not retain method=mtp; got "
                f"{speculative_method!r}"
            )
        if speculative_tokens != MTP_SPECULATIVE_TOKENS:
            raise RuntimeError(
                "vLLM did not retain MTP3; got num_speculative_tokens="
                f"{speculative_tokens!r}"
            )
        if speculative_enforce_eager != mtp_drafter_enforce_eager:
            raise RuntimeError(
                "vLLM did not retain the requested MTP drafter eager mode; "
                f"got {speculative_enforce_eager!r}"
            )
    elif speculative_config is not None:
        raise RuntimeError(
            "vLLM unexpectedly enabled speculative decoding for the "
            "autoregressive baseline"
        )


def main() -> None:
    _ensure_project_is_importable()

    prompt_lengths = parse_prompt_lengths()
    batch_size = len(prompt_lengths)
    tp_size = _tp_size()
    enable_mtp = env_bool("VLLM_ENABLE_MTP", False)
    num_speculative_tokens = (
        MTP_SPECULATIVE_TOKENS if enable_mtp else 0
    )
    decode_tokens_per_request = 1 + num_speculative_tokens
    capture_size = batch_size * decode_tokens_per_request
    max_gen_tokens = _max_gen_tokens(enable_mtp)
    max_model_len = max(prompt_lengths) + max_gen_tokens
    max_num_batched_tokens = env_int(
        "VLLM_MAX_NUM_BATCHED_TOKENS",
        DEFAULT_MAX_NUM_BATCHED_TOKENS,
    )
    enable_profile = env_bool("VLLM_ENABLE_PROFILE", True)
    mtp_drafter_enforce_eager = env_bool(
        "VLLM_MTP_DRAFTER_ENFORCE_EAGER",
        True,
    )
    quantization = os.environ.get("VLLM_QUANTIZATION")
    if enable_mtp and quantization is None:
        quantization = "ascend"
    additional_config = env_json_object(
        "VLLM_ADDITIONAL_CONFIG",
        DEFAULT_GLM_MTP_ADDITIONAL_CONFIG if enable_mtp else {},
    )
    assert additional_config is not None
    target_rank = env_int("VLLM_PROFILE_GLOBAL_RANK", 0)
    profile_dir = (
        _profile_dir(tp_size, batch_size, enable_mtp)
        if enable_profile
        else None
    )
    prefill_chunk_tokens = env_int(
        "VLLM_PREFILL_CHUNK_TOKENS",
        max(1, max_num_batched_tokens // batch_size),
    )

    minimum_gen_tokens = (
        MTP_SPECULATIVE_TOKENS + 2 if enable_mtp else 2
    )
    if max_gen_tokens < minimum_gen_tokens:
        raise ValueError(
            "VLLM_MAX_GEN_TOKENS must be at least "
            f"{minimum_gen_tokens} so that a complete "
            f"{'MTP3 verification' if enable_mtp else 'decode'} step exists"
        )
    if max_num_batched_tokens < capture_size:
        raise ValueError(
            "VLLM_MAX_NUM_BATCHED_TOKENS must be >= the full decode "
            f"verification size ({capture_size})"
        )
    if prefill_chunk_tokens <= 0:
        raise ValueError("VLLM_PREFILL_CHUNK_TOKENS must be positive")
    if not 0 <= target_rank < tp_size:
        raise ValueError(
            f"VLLM_PROFILE_GLOBAL_RANK must be in [0, {tp_size}), got "
            f"{target_rank}"
        )
    if env_bool("VLLM_ENFORCE_EAGER", False):
        raise ValueError(
            "profile_vllm.py is the FULL_DECODE_ONLY graph baseline; "
            "VLLM_ENFORCE_EAGER must be 0"
        )

    if profile_dir is not None:
        if profile_dir.exists() and any(profile_dir.iterdir()):
            raise RuntimeError(
                f"VLLM_PROFILE_DIR must be empty, got non-empty directory: "
                f"{profile_dir}"
            )
        profile_dir.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_PROFILE_EXPECTED_BATCH_SIZE"] = str(batch_size)
    os.environ["VLLM_PROFILE_GLOBAL_RANK"] = str(target_rank)
    os.environ[PROFILE_SPECULATIVE_TOKENS_ENV] = str(
        num_speculative_tokens
    )

    # Imported after PYTHONPATH and profiler-control environment are finalized;
    # spawned TP workers inherit both values.
    from vllm import LLM, SamplingParams, TokensPrompt

    from vllm_decode_profile.profile_prompts import (
        build_base_meaningful_prompt,
        build_exact_token_prompt,
        decode_prompt_tail,
    )

    model_path = _model_path()
    enable_expert_parallel = env_bool(
        "VLLM_ENABLE_EXPERT_PARALLEL",
        True,
    )
    decode_mode = (
        f"mtp{num_speculative_tokens}" if enable_mtp else "decode"
    )
    profile_prefix = (
        f"vllm_baseline_tp{tp_size}_bs{batch_size}_"
        f"seq{min(prompt_lengths)}_{max(prompt_lengths)}_{decode_mode}_"
        "full_decode_only"
    )

    print(
        "vLLM decode run config: "
        f"model={model_path}, tp={tp_size}, ep={enable_expert_parallel}, "
        f"batch={batch_size}, prompt_lengths={prompt_lengths}, "
        f"max_model_len={max_model_len}, max_gen_tokens={max_gen_tokens}, "
        f"max_num_batched_tokens={max_num_batched_tokens}, "
        f"per_request_prefill_chunk_cap={prefill_chunk_tokens}, "
        f"decode_mode={decode_mode}, capture_size={capture_size}, "
        f"quantization={quantization}, "
        "mtp_drafter_enforce_eager="
        f"{mtp_drafter_enforce_eager if enable_mtp else 'n/a'}, "
        f"additional_config={additional_config}, "
        f"all2all_backend={BASELINE_ALL2ALL_BACKEND}, "
        f"profile_enabled={enable_profile}, "
        f"profile_global_rank={target_rank}, "
        f"profile_dir={profile_dir if profile_dir is not None else 'disabled'}"
    )
    print("runtime: native vLLM + vLLM-Ascend baseline")

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        enable_expert_parallel=enable_expert_parallel,
        quantization=quantization,
        additional_config=additional_config,
        # vLLM-Ascend normally selects this while resolving worker_cls="auto".
        # Our profiling-only worker subclass bypasses that branch, so preserve
        # the baseline backend explicitly.
        all2all_backend=BASELINE_ALL2ALL_BACKEND,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=batch_size,
        gpu_memory_utilization=env_float(
            "VLLM_GPU_MEMORY_UTILIZATION",
            0.95,
        ),
        block_size=env_int(
            "VLLM_KVCACHE_BLOCK_SIZE",
            128,
        ),
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        # vLLM 0.19 rejects non-default Concurrent Partial Prefill knobs.
        # Keeping their defaults (both 1) does not serialize this V1
        # scheduler: the shared token budget plus this per-request threshold
        # still lets all requests advance one chunk in the same engine step.
        long_prefill_token_threshold=prefill_chunk_tokens,
        async_scheduling=False,
        scheduler_cls=SCHEDULER_CLASS,
        trust_remote_code=True,
        enforce_eager=False,
        worker_cls=WORKER_CLASS,
        speculative_config=(
            {
                "method": "deepseek_mtp",
                "num_speculative_tokens": MTP_SPECULATIVE_TOKENS,
                "enforce_eager": mtp_drafter_enforce_eager,
            }
            if enable_mtp
            else None
        ),
        compilation_config={
            "mode": "VLLM_COMPILE",
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [capture_size],
        },
        profiler_config=(
            {
                "profiler": "torch",
                "torch_profiler_dir": str(profile_dir),
                "torch_profiler_with_stack": False,
                "torch_profiler_record_shapes": False,
                "torch_profiler_with_memory": False,
            }
            if enable_profile
            else None
        ),
    )
    _print_effective_runtime_config(
        llm,
        enable_expert_parallel,
        batch_size,
        enable_mtp,
        mtp_drafter_enforce_eager,
    )

    tokenizer = llm.get_tokenizer()
    base_ids = build_base_meaningful_prompt(tokenizer)
    prompt_token_ids = [
        build_exact_token_prompt(base_ids, length)
        for length in prompt_lengths
    ]
    token_prompts = [
        TokensPrompt(prompt_token_ids=token_ids)
        for token_ids in prompt_token_ids
    ]

    print(f"meaningful_base_tokens={len(base_ids)}")
    for index, token_ids in enumerate(prompt_token_ids, 1):
        print(
            f"prompt {index}: token_len={len(token_ids)}, "
            f"first_ids={token_ids[:16]}, "
            f"tail={decode_prompt_tail(tokenizer, token_ids)!r}"
        )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_gen_tokens,
        min_tokens=max_gen_tokens,
        ignore_eos=True,
    )

    if enable_profile:
        trigger_description = (
            "MTP3 verification (1 ordinary + 3 draft tokens per request)"
            if enable_mtp
            else "pure single-token decode"
        )
        print(
            "profiling is armed now; "
            f"rank {target_rank} starts recording only when the full batch "
            f"reaches {trigger_description}"
        )
    else:
        print(
            "profiling is disabled; decode-step statistics remain enabled"
        )
    outputs = None
    profile_armed = False
    try:
        if enable_profile:
            llm.start_profile(profile_prefix=profile_prefix)
            profile_armed = True
        outputs = llm.generate(
            token_prompts,
            sampling_params,
            use_tqdm=False,
        )
    finally:
        if profile_armed:
            llm.stop_profile()

    if enable_profile:
        flush_seconds = env_float("VLLM_PROFILE_FLUSH_SECONDS", 10.0)
        if flush_seconds > 0:
            time.sleep(flush_seconds)

    assert outputs is not None
    for index, (token_ids, output) in enumerate(
        zip(prompt_token_ids, outputs),
        1,
    ):
        completion = output.outputs[0]
        print(f"prompt {index}: prompt_len={len(token_ids)}")
        print("response  :", repr(completion.text))
        print("token_ids :", completion.token_ids)
        print()

    if enable_profile:
        print(f"profile saved to: {profile_dir}")
        print(
            "required worker markers: VLLM_BASELINE_PROFILE_STARTED and "
            "VLLM_BASELINE_PROFILE_STOPPED"
        )
    else:
        print("profile disabled by VLLM_ENABLE_PROFILE=0")


if __name__ == "__main__":
    main()
