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

## 运行profile采集

```bash
export VLLM_MODEL=/mnt/models/GLM-5.1-w4a8/         # set model path here
export VLLM_TP_SIZE=16
export VLLM_ENABLE_EXPERT_PARALLEL=1
export VLLM_QUANTIZATION=ascend
export VLLM_MAX_NUM_BATCHED_TOKENS=4096
export VLLM_GPU_MEMORY_UTILIZATION=0.88
export VLLM_KVCACHE_BLOCK_SIZE=128
export VLLM_MTP_DRAFTER_ENFORCE_EAGER=0
export VLLM_ENABLE_MTP=1                            # set 0 to disable MTP 
export VLLM_PROMPT_LENGTHS=40000,40001,40002,40003  # seqlen=40k, bs=4 
export VLLM_MAX_GEN_TOKENS=40                       # decode enough tokens
export VLLM_PROFILE_GLOBAL_RANK=0                   # only profile rank0
export VLLM_ENABLE_PROFILE=1                        # set to 0 will only print decode step time, without profile output 
export VLLM_PROFILE_DIR=$PWD/vllm_seq40k_bs4_mtp    # set output profile path here 
PYTHONPATH=$PWD:$PYTHONPATH python3 profile_vllm.py
python parse_profile.py
```


