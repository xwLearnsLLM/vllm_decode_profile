# vLLM Decode Profile

该仓库用于采集 vLLM 0.19 + vLLM-Ascend 0.19 的纯 decode profile，并打印
decode step 的本地 batch size、TPOT、序列长度和 KV Cache 占用。支持两种模式：

- 离线 `LLM.generate()` 的 TP-only 基线；
- 在线 `vllm serve` 的双机 DP32 + EP32 + TP1，参见
  [`README_DP32.md`](README_DP32.md)。

脚本具有以下约束：

- 使用 `FULL_DECODE_ONLY`，只 capture 实际需要的 decode batch size。
- profiler 只在指定的 `(DP rank, TP rank)` worker 上创建；默认是
  `DP rank 0 / TP rank 0`。
- decode step 统计只在指定 DP EngineCore 上进行；其余 DP rank 不计时、不缓存、
  不打印。
- chunk prefill、混合 prefill/decode、spec decode、encoder step 以及不符合目标
  batch size 的 step 都不会触发 profiler。
- profiler 启动后，默认在 decode 阶段发生变化时、达到最大 decode step 数时，
  或收到 `stop_profile()` 时停止。
- profiling worker 只改变采集控制。模型执行、KV Cache 和 kernel 仍来自原生
  `NPUWorker`。

## 环境要求

目标环境需要已经安装匹配版本的 vLLM、vLLM-Ascend 和 torch-npu。双机在线
环境的逐项检查参见 [`README_DP32.md`](README_DP32.md)。

## 离线 TP-only 用法

在仓库根目录配置：

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

export VLLM_MODEL=/home/models/DeepSeek-V3.2-REAP-345B-A37B-BF16/
export VLLM_TP_SIZE=16
export VLLM_ENABLE_EXPERT_PARALLEL=1
export VLLM_KVCACHE_BLOCK_SIZE=128
export VLLM_ENFORCE_EAGER=0
export VLLM_ENABLE_PROFILE=1

# 推荐 8～16，以便采到多个稳定 decode step。
export VLLM_MAX_GEN_TOKENS=8

# vLLM chunk-prefill 每个 EngineCore step 的总 token budget。
export VLLM_MAX_NUM_BATCHED_TOKENS=4096
export VLLM_GPU_MEMORY_UTILIZATION=0.95

# 离线 TP-only 兼容参数：采集 TP rank 0。
export VLLM_PROFILE_GLOBAL_RANK=0
```

例如采集 BS=6、输入长度约 25k：

```bash
export VLLM_PROMPT_LENGTHS=25000,25001,25002,25003,25004,25005
```

只看 decode 统计：

```bash
VLLM_ENABLE_PROFILE=0 PYTHONPATH=$PWD:$PYTHONPATH python3 profile_vllm.py
```

同时采集 profile：

```bash
export VLLM_PROFILE_DIR=$PWD/profiles/with_profile_$(date +%Y%m%d_%H%M%S)
VLLM_ENABLE_PROFILE=1 PYTHONPATH=$PWD:$PYTHONPATH python3 profile_vllm.py
```

## 在线双机 DP32 + EP32 + TP1

固定拓扑、预设启动脚本、在线请求、单卡 profiling 和故障检查已经集中到
[`README_DP32.md`](README_DP32.md)。

## 解析 profile

在实际生成 profile 的节点上运行：

```bash
python3 parse_profile.py
```

脚本会在参数指定的目录或 `VLLM_PROFILE_DIR` 下找到唯一的 `*ascend_pt` 原始
目录；尚未解析时调用 `torch_npu.profiler.profiler.analyse()`，已经存在
`ASCEND_PROFILER_OUTPUT` 时直接复用，并打印 `trace_view.json` 路径。然后可用
MindStudio Insight 打开该文件。

默认仅采一张卡，因此解析脚本仍要求只有一个 `*ascend_pt`。

## Decode-step 输出

日志仅来自选中的 Scheduler，例如：

```text
VLLM_DECODE_STEP_LOG_BEGIN dp_rank=0 count=2 reason=idle
[VLLM_DECODE step=0001 dp_rank=0] bsz=8, num_tokens=8, TPOT=68.420 ms, seq_lens=[...], HBM_KV=1407/1680 blocks (83.75%)
[VLLM_DECODE step=0002 dp_rank=0] bsz=8, num_tokens=8, TPOT=67.981 ms, seq_lens=[...], HBM_KV=1407/1680 blocks (83.75%)
VLLM_DECODE_STEP_LOG_END
```

这些指标都是目标 EngineCore 的本地数据：

- `bsz`、`num_tokens` 和 `seq_lens` 不是跨 EngineCore 的汇总值；
- `HBM_KV` 是目标 NPU 的本地 KV block 占用；
- `TPOT` 是从 Scheduler 开始调度到处理完模型输出的 EngineCore step 墙钟时间。

## 离线常用参数

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `VLLM_ENABLE_PROFILE` | `1` | 离线入口专用；`0` 只打印 decode 统计 |
| `VLLM_PROFILE_DIR` | 带时间戳的 `profiles/...` | 离线 profiler 输出目录 |
| `VLLM_PROFILE_GLOBAL_RANK` | `0` | 离线 TP-only 兼容参数 |
| `VLLM_MAX_NUM_BATCHED_TOKENS` | `8192` | 离线 chunk-prefill 每步总 token budget |
| `VLLM_PREFILL_CHUNK_TOKENS` | 总 budget 除以 BS | 离线单请求每步 prefill 上限 |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.95` | 离线 vLLM NPU 显存利用率 |
| `VLLM_PROFILE_FLUSH_SECONDS` | `10` | 离线 `stop_profile()` 后等待文件落盘的秒数 |

## 离线必查日志

离线入口初始化后应看到：

```text
effective runtime config: cudagraph_mode=...FULL_DECODE_ONLY..., capture_sizes=[6], all2all_backend=flashinfer_all2allv, scheduler_cls=vllm_decode_profile.profile_scheduler.DecodeStepLoggingScheduler
runtime: native vLLM + vLLM-Ascend baseline
```

离线脚本不会设置 `max_num_partial_prefills` 或 `max_long_partial_prefills`。vLLM
0.19 会拒绝非默认值并报 `Concurrent Partial Prefill is not supported`；这里通过
`max_num_batched_tokens` 和 `long_prefill_token_threshold` 让同一批请求按 chunk
同步推进，最终形成完整 batch 的纯 decode step。
