# vLLM Decode Profile

该仓库用于采集 vLLM 0.19 + vLLM-Ascend 0.19 的 TP rank0 decode step 的 profile ，大大降低 profile 文件的大小。并顺手打印 decode step 的 batchsize, TPOT 等信息

脚本具有以下约束：

- 强制使用 `FULL_DECODE_ONLY`，只 capture 实际 batch size。
- profiler 只在指定 global rank 上创建；TP-only 场景下 global rank 0 就是 TP rank 0。
- profiling worker 仅用于控制采集窗口；EP 通信后端显式保持为 vLLM-Ascend baseline 默认选择的 `flashinfer_all2allv`。
- `start_profile()` 只负责武装 profiler。worker 观察到完整 batch 首次进入 纯 single-token decode 时才真正开始采集，程序生成结束时停止。
- chunk prefill、混合 prefill/decode 以及不完整 decode batch 都不会触发 profiler；首次完整 batch 纯 decode 触发后，会持续采集到本次生成结束。
- 纯 single-token decode step 会记录 BS、token 数、EngineCore TPOT、各请求当前序列长度和逻辑 HBM KV-block 使用量；所有记录在请求结束后一次性打印，避免 stdout 放大相邻 decode step 的空隙。

　

## 环境要求

仓库运行时只需要目标环境已经安装 vLLM 和 vLLM-Ascend。

　

## 使用方法举例：TP16、BS6、30K序列长度的信息采集

在仓库根目录，先做一些基础配置：

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

# vLLM chunk-prefill 每个 engine step 的总 token budget。
export VLLM_MAX_NUM_BATCHED_TOKENS=4096
export VLLM_GPU_MEMORY_UTILIZATION=0.95

# 设置成只采集 TP rank 0
export VLLM_PROFILE_GLOBAL_RANK=0
```

设置你想采集的测试用例，例如如果你想采集 bs=6, seqlen=25k ，则这样设置：

```bash
export VLLM_PROMPT_LENGTHS=25000,25001,25002,25003,25004,25005
```

然后运行推理。如果不想采集profile，只想看每个 decode-step 的 batchsize、TPOT等信息：

```bash
VLLM_ENABLE_PROFILE=0 PYTHONPATH=$PWD:$PYTHONPATH python3 profile_vllm.py
```

如果想采 profile：

```bash
export VLLM_PROFILE_DIR=$PWD/profiles/with_profile_$(date +%Y%m%d_%H%M%S)
VLLM_ENABLE_PROFILE=1 PYTHONPATH=$PWD:$PYTHONPATH python3 profile_vllm.py
```

　

常用参数如下：

| 参数                          |                    默认值 | 说明                                                         |
| ----------------------------- | ------------------------: | ------------------------------------------------------------ |
| `VLLM_ENABLE_PROFILE`         |                       `1` | `0`：只打印 decode 统计；`1`：同时采集 profile              |
| `VLLM_PROFILE_DIR`            | 带时间戳的 `profiles/...` | profile 输出目录，运行前会自动创建。必须是不存在或为空的目录，避免多次结果混在一起。 |
| `VLLM_PROFILE_GLOBAL_RANK`    |                       `0` | 唯一采集的 global rank；TP-only 时等同 TP rank               |
| `VLLM_MAX_NUM_BATCHED_TOKENS` |                    `8192` | chunk-prefill 每步总 token budget                            |
| `VLLM_PREFILL_CHUNK_TOKENS`   |         总 budget 除以 BS | 单请求每步 prefill 上限                                      |
| `VLLM_GPU_MEMORY_UTILIZATION` |                    `0.95` | vLLM NPU 显存利用率                                          |
| `VLLM_PROFILE_FLUSH_SECONDS`  |                      `10` | `stop_profile()` 后等待文件落盘的秒数                        |

脚本不会设置 `max_num_partial_prefills` 或 `max_long_partial_prefills`。vLLM
0.19 会拒绝非默认值并报 `Concurrent Partial Prefill is not supported`；这里通过
`max_num_batched_tokens` 和 `long_prefill_token_threshold` 让同一批请求在 V1
scheduler 中按 chunk 同步推进，最终仍会形成完整 batch 的纯 decode step。

　

## 必须检查的运行日志

初始化后必须看到：

```text
effective runtime config: cudagraph_mode=...FULL_DECODE_ONLY..., capture_sizes=[6], all2all_backend=flashinfer_all2allv, scheduler_cls=vllm_decode_profile.profile_scheduler.DecodeStepLoggingScheduler
runtime: native vLLM + vLLM-Ascend baseline
```

`VLLM_ENABLE_PROFILE=1` 时，profile 控制必须依次出现：

```text
VLLM_BASELINE_PROFILE_ARMED rank=0 expected_batch=6
VLLM_BASELINE_PROFILE_STARTED rank=0 batch=6 phase=decode
VLLM_BASELINE_PROFILE_STOPPED rank=0
```

只有 rank 0 会真正创建 profiler，因此输出目录中正常情况下只有带 `tp0` 的 `*ascend_pt` 数据。如果出现 `VLLM_BASELINE_PROFILE_NOT_STARTED` ，说明运行期间没有形成 BS6 的纯 single-token decode step。优先检查 prompt 长度是否过于悬殊、生成 token 是否太少，以及 KV cache 是否足以让全部请求同时驻留。

`VLLM_ENABLE_PROFILE=0` 时不会出现上述 profile marker，而会打印：

```text
profiling is disabled; decode-step statistics remain enabled
profile disabled by VLLM_ENABLE_PROFILE=0
```

　

## Decode-step 打印信息统计

生成结束时会一次性输出本轮所有纯 single-token decode step，例如：

```text
[VLLM_DECODE step=0001] bsz=6, num_tokens=6, TPOT=68.420 ms, seq_lens=[30001, 30002, 30003, 30004, 30005, 30006], HBM_KV=1407/1680 blocks (83.75%)
[VLLM_DECODE step=0002] bsz=6, num_tokens=6, TPOT=67.981 ms, seq_lens=[30002, 30003, 30004, 30005, 30006, 30007], HBM_KV=1407/1680 blocks (83.75%)
```

　

## 解析和查看 profile

如果你在推理时设置了 `VLLM_ENABLE_PROFILE=1` 和 `VLLM_PROFILE_DIR` ，则可以输出 profile ，并用 

先找到唯一的 rank-0 原始目录：

```bash
find "$VLLM_PROFILE_DIR" -type d -name '*ascend_pt'
```

如果目录下还没有 `ASCEND_PROFILER_OUTPUT`，手动执行分析：

```bash
RAW_PROFILE_DIR=$(find "$VLLM_PROFILE_DIR" -type d -name '*ascend_pt' | head -n 1)

python3 - "$RAW_PROFILE_DIR" <<'PY'
import sys
from torch_npu.profiler.profiler import analyse

analyse(sys.argv[1])
PY
```

分析完成后查找：

```bash
find "$RAW_PROFILE_DIR" -path '*/ASCEND_PROFILER_OUTPUT/trace_view.json'
```

使用 MindStudio Insight 打开 `trace_view.json` 或其所在的`ASCEND_PROFILER_OUTPUT` 目录。重点观察中间几个稳定 decode step；首次 decode step 可能包含 profiler 启动开销，不用于 TPOT 对比。

　
