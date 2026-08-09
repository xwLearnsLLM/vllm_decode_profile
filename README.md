# vLLM Decode Profile

该仓库用于采集 vLLM 0.19.1 + vLLM-Ascend v0.19.1rc1（0.19.1 系列）的 TP rank0 decode profile，并在生成结束后打印每个纯 decode step 的 batch size、时延和 KV cache 占用。支持两种模式：

- 普通自回归 decode；
- GLM-5/GLM-5.1 的 MTP3，即每个请求每步包含 1 个普通 token 和 3 个猜测 token。

脚本具有以下约束：

- target model 强制使用 `FULL_DECODE_ONLY`。普通 decode capture `BS` 个 token；MTP3 target verification capture `BS * 4` 个 token。
- profiler 只在指定 global rank 上创建；TP-only 场景下 global rank 0 就是 TP rank 0。
- profiling worker 仅用于控制采集窗口；EP 通信后端显式保持为 vLLM-Ascend baseline 的 `flashinfer_all2allv`。
- chunk prefill、混合 prefill/decode、不完整 decode batch，以及 MTP 尚未产生 3 个 draft 的 bootstrap step，都不会触发 profiler。首次完整 batch 纯 decode/MTP3 verification 触发后，会持续采集到本次生成结束。

## 环境与兼容性

MTP3 实现按以下官方版本和配置核对：

- [vLLM-Ascend v0.19.1rc1](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.19.1rc1) 对应 vLLM v0.19.1；该版本发布说明包含 MTP merged graph 和融合 W4A8 kernel。
- 官方 [GLM-5/GLM-5.1 指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/GLM5.html) 列出 `GLM-5.1-w4a8`，并给出 Atlas 800 A3、TP16、`quantization=ascend`、`deepseek_mtp`、`num_speculative_tokens=3` 的配置。
- 对应软件矩阵为 Python 3.10/3.11、CANN 8.5.1、PyTorch/torch_npu 2.9.0 和 Triton-Ascend 3.2.0；该 release 的 Known Issues 另建议把 torch_npu 升到 `2.9.0.post1+git4c901a4`，实际安装应以 release 说明为准。

因此，从 0.19.1 源码和官方配置看，GLM-5.1-w4a8 在 Ascend 910C/Atlas A3 软件栈上具备 MTP3 支持。这里的自动化测试不包含真实 NPU，最终仍需在目标 910C 机器上确认启动、输出正确性和接受率。

该 release 还记录了 GLM-5/GLM-5.1 单机运行错误/结果异常的[已知问题 #8843](https://github.com/vllm-project/vllm-ascend/issues/8843)，而当前官方 GLM 指南也明确要求 MTP drafter 使用 `enforce_eager=true`。因此本仓库默认 `VLLM_MTP_DRAFTER_ENFORCE_EAGER=1`；target model 仍保持 `FULL_DECODE_ONLY`，rank0 profile 仍覆盖完整 MTP decode step。只有确认本机组合稳定后，才建议设为 `0` 试验 merged-graph drafter。

## 普通 decode 配置示例

在仓库根目录设置：

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

export VLLM_MODEL=/home/models/DeepSeek-V3.2-REAP-345B-A37B-BF16/
export VLLM_TP_SIZE=16
export VLLM_ENABLE_EXPERT_PARALLEL=1
export VLLM_KVCACHE_BLOCK_SIZE=128
export VLLM_ENFORCE_EAGER=0
export VLLM_ENABLE_MTP=0
export VLLM_ENABLE_PROFILE=1

# 推荐 8～16，以便采到多个稳定 decode step。
export VLLM_MAX_GEN_TOKENS=8

# vLLM chunk-prefill 每个 engine step 的总 token budget。
export VLLM_MAX_NUM_BATCHED_TOKENS=4096
export VLLM_GPU_MEMORY_UTILIZATION=0.95

# 只采集 TP rank 0。
export VLLM_PROFILE_GLOBAL_RANK=0
```

设置测试用例。例如采集 BS=6、sequence length 约 25k：

```bash
export VLLM_PROMPT_LENGTHS=25000,25001,25002,25003,25004,25005
```

## GLM-5.1-w4a8 MTP3 配置示例

在上述通用环境变量基础上改为：

```bash
export VLLM_MODEL=/home/models/GLM-5.1-w4a8/
export VLLM_TP_SIZE=16
export VLLM_ENABLE_EXPERT_PARALLEL=1
export VLLM_ENABLE_MTP=1
export VLLM_QUANTIZATION=ascend

# 推荐的安全默认值：MTP drafter eager，target decode 仍用图。
export VLLM_MTP_DRAFTER_ENFORCE_EAGER=1

# 确保能形成多个完整的 MTP3 verification step。
export VLLM_MAX_GEN_TOKENS=16

export VLLM_ADDITIONAL_CONFIG='{"fuse_muls_add":true,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"enable_npugraph_ex":true}}'
```

开启 MTP 时，如果没有显式设置，脚本会自动采用上面的 `VLLM_QUANTIZATION=ascend`、`VLLM_ADDITIONAL_CONFIG`、MTP3 和 drafter eager 配置。当前只支持 MTP3，不提供可变 speculative-token 数量。

## 运行

只看 decode-step 信息、不采集 profile：

```bash
VLLM_ENABLE_PROFILE=0 PYTHONPATH=$PWD:$PYTHONPATH python3 profile_vllm.py
```

采集 profile：

```bash
export VLLM_PROFILE_DIR=$PWD/profiles/with_profile_$(date +%Y%m%d_%H%M%S)
VLLM_ENABLE_PROFILE=1 PYTHONPATH=$PWD:$PYTHONPATH python3 profile_vllm.py
```

导出 profile：

```bash
python3 parse_profile.py
```

该脚本会自动找到唯一的 rank0 `*ascend_pt` 原始目录；尚未解析时调用 `torch_npu.profiler.profiler.analyse()`，已经存在 `ASCEND_PROFILER_OUTPUT` 时则直接复用，并打印输出目录及 `trace_view.json` 路径。之后可把 `trace_view.json` 下载到本地，用 MindStudio Insight 查看。

## Decode-step 打印信息

不开 MTP 时，输出格式和原行为保持一致：

```text
[VLLM_DECODE step=0001] bsz=6, num_tokens=6, TPOT=68.420 ms, seq_lens=[30001, 30002, 30003, 30004, 30005, 30006], HBM_KV=1407/1680 blocks (83.75%)
```

开启 MTP3 时：

```text
[VLLM_DECODE step=0001] bsz=6, num_tokens=6, STEP=80.000 ms, accept/draft=10/18, seq_lens=[30003, 30004, 30002, 30005, 30003, 30005], HBM_KV=1407/1680 blocks (83.75%)
```

- `STEP`：从 scheduler 完成该轮调度，到对应 `ModelRunnerOutput` 返回的墙钟时延；它覆盖一次完整 MTP `execute_model`/forward 临界路径，不包含该轮 `schedule()` 和 `update_from_output()` 的主要开销，也不是单个 NPU kernel event 的时间。
- `accept/draft`：`accept` 是全部请求接受的猜测 token 总数，不包含每个请求必然输出的普通 token；`draft` 固定为 `BS * 3`。上例即接受 10 个，共投机 18 个。
- MTP 模式下的 `seq_lens` 是拒绝 draft 回滚之后的有效 sequence length，因此不同请求可按各自接受数量增长。

日志会在本轮请求全部结束时一次性输出，避免逐 step 的 stdout 干扰下一步时延。

## 附录 A：常用参数

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `VLLM_ENABLE_MTP` | `0` | `0`：普通 decode；`1`：固定启用 MTP3 |
| `VLLM_MTP_DRAFTER_ENFORCE_EAGER` | `1` | MTP drafter 是否 eager；target model 始终使用 `FULL_DECODE_ONLY` |
| `VLLM_QUANTIZATION` | MTP 时为 `ascend` | 传给 vLLM 的 quantization 配置 |
| `VLLM_ADDITIONAL_CONFIG` | MTP 时使用 GLM 推荐配置 | JSON object，传给 vLLM `additional_config` |
| `VLLM_ENABLE_PROFILE` | `1` | `0`：只打印 decode 统计；`1`：同时采集 profile |
| `VLLM_PROFILE_DIR` | 带时间戳的 `profiles/...` | profile 输出目录；必须不存在或为空 |
| `VLLM_PROFILE_GLOBAL_RANK` | `0` | 唯一采集的 global rank；TP-only 时等同 TP rank |
| `VLLM_MAX_GEN_TOKENS` | 普通 `8`；MTP `16` | 每个请求固定生成的 token 上限 |
| `VLLM_MAX_NUM_BATCHED_TOKENS` | `8192` | chunk-prefill 和 decode verification 每步总 token budget；MTP3 至少为 `BS * 4` |
| `VLLM_PREFILL_CHUNK_TOKENS` | 总 budget 除以 BS | 单请求每步 prefill 上限 |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.95` | vLLM NPU 显存利用率 |
| `VLLM_PROFILE_FLUSH_SECONDS` | `10` | `stop_profile()` 后等待文件落盘的秒数 |

脚本不会设置 `max_num_partial_prefills` 或 `max_long_partial_prefills`。vLLM 0.19.1 会拒绝非默认值并报 `Concurrent Partial Prefill is not supported`；这里通过 `max_num_batched_tokens` 和 `long_prefill_token_threshold` 让同一批请求在 V1 scheduler 中按 chunk 同步推进，最终仍形成完整 batch 的纯 decode step。

## 附录 B：运行日志检查

普通 BS6 decode 初始化后必须看到类似：

```text
effective runtime config: cudagraph_mode=...FULL_DECODE_ONLY..., capture_sizes=[6], all2all_backend=flashinfer_all2allv, scheduler_cls=vllm_decode_profile.profile_scheduler.DecodeStepLoggingScheduler, speculative_method=None, speculative_tokens=0, speculative_enforce_eager=None
runtime: native vLLM + vLLM-Ascend baseline
```

MTP3 BS6 必须看到 `capture_sizes` 包含 24，并确认 MTP3 生效：

```text
effective runtime config: cudagraph_mode=...FULL_DECODE_ONLY..., capture_sizes=[24], all2all_backend=flashinfer_all2allv, scheduler_cls=vllm_decode_profile.profile_scheduler.DecodeStepLoggingScheduler, speculative_method=mtp, speculative_tokens=3, speculative_enforce_eager=True
```

`VLLM_ENABLE_PROFILE=1` 且 MTP3 BS6 时，profile 控制应依次出现：

```text
VLLM_BASELINE_PROFILE_ARMED rank=0 expected_batch=6 speculative_tokens=3
VLLM_BASELINE_PROFILE_STARTED rank=0 batch=6 phase=mtp3_decode scheduled_tokens=24
VLLM_BASELINE_PROFILE_STOPPED rank=0
```

不开 MTP 时对应 `speculative_tokens=0`、`phase=decode`、`scheduled_tokens=6`。只有指定 rank 会真正创建 profiler；默认 rank0，因此输出目录正常情况下只有带 `tp0` 的 `*ascend_pt` 数据。

如果出现 `VLLM_BASELINE_PROFILE_NOT_STARTED`，说明运行期间没有形成完整 batch 的目标 decode step。优先检查 prompt 长度差异、生成 token 数、`VLLM_MAX_NUM_BATCHED_TOKENS` 和 KV cache 容量。

`VLLM_ENABLE_PROFILE=0` 时不会出现上述 profile marker，而会打印：

```text
profiling is disabled; decode-step statistics remain enabled
profile disabled by VLLM_ENABLE_PROFILE=0
```
