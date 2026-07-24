# 双机 DP32 + EP32 在线推理与 Decode Profiling

本文档适用于下面这套固定环境，并与 `run_dp32/master.sh`、
`run_dp32/headless.sh` 中的默认值一致。只要两台机器已经安装好匹配版本的
vLLM、vLLM-Ascend 和 torch-npu，并且都放置了本仓库，就不需要再手工拼接
`vllm serve` 参数。

## 预设拓扑

| 角色 | IP | 本地 NPU | 全局 DP ranks | 启动脚本 |
| --- | --- | ---: | --- | --- |
| master/API 节点 | `10.44.53.212` | 16 张，设备 0-15 | 0-15 | `run_dp32/master.sh` |
| headless 节点 | `10.44.53.215` | 16 张，设备 0-15 | 16-31 | `run_dp32/headless.sh` |

两台机器的模型目录均预设为：

```text
/mnt/models/GLM-5.1-w4a8
```

其余关键预设如下：

| 配置 | 预设值 |
| --- | --- |
| 并行方式 | DP32 + EP32 + TP1 |
| 对外服务地址 | `http://10.44.53.212:8077` |
| DP coordinator | `10.44.53.212:12890` |
| API 中的模型名 | `glm-5` |
| 每个节点的本地 DP 数 | 16 |
| `max_num_seqs` | 每个 DP rank 16 |
| `max_model_len` | 20480 |
| `max_num_batched_tokens` | 16384 |
| KV block size | 128 |
| graph capture sizes | 1、2、4、8 |
| all-to-all 后端 | `flashinfer_all2allv` |

该拓扑使用 vLLM 的多机 internal DP 方式：master 同时运行 API server 和
DP ranks 0-15，另一台机器以 `--headless` 方式运行 DP ranks 16-31。

## Profiling 预设

默认只在 master 节点上采集一张卡：

- profile 目标是全局 `DP rank 0 / TP rank 0`；
- decode step 日志只由 `DP rank 0` 的 Scheduler 打印；
- 其余 31 个 DP rank 正常参与 DP32 + EP32 推理，但不创建 profiler；
- 当 DP0 出现本地 BS=8 的纯 single-token decode step 时开始采集；
- 最多采集 8 个连续 decode step；
- decode 阶段发生变化时，在执行非目标 step 之前停止采集。

这里的 BS=8 是 `DP rank 0` 的本地 batch size，不是整个 DP32 服务的全局
batch size。

两个节点都必须使用仓库中的自定义 Worker。master 会把 profile 控制广播给所有
DP EngineCore；非目标 worker 收到命令后会立即返回。脚本还显式指定了
`flashinfer_all2allv`，避免自定义 Worker 绕过 vLLM-Ascend 的自动后端选择。

## 1. 两台机器启动前检查

仓库在两台机器上的绝对路径可以不同，因为启动脚本会根据自身位置自动设置
`PYTHONPATH`。在两台机器上分别进入各自的仓库根目录，然后执行：

```bash
test -d /mnt/models/GLM-5.1-w4a8 && echo "model directory: OK"
command -v vllm
vllm --version
python3 -c 'import vllm, vllm_ascend, torch_npu; print("Python imports: OK")'
npu-smi info
```

还需要确认：

- `10.44.53.212` 确实配置在 master 的推理网卡上；
- `10.44.53.215` 确实配置在 headless 节点的推理网卡上；
- 两台机器之间网络互通，防火墙不会拦截 HCCL 通信；
- master 的 TCP 端口 `12890` 可用于 DP coordinator；
- master 的 TCP 端口 `8077` 可用于 API server。

可分别检查本机 IP：

```bash
ip -o -4 addr show
```

脚本会根据预设 IP 自动找到对应的网卡名，并将其写入 HCCL、Gloo 和 TP 的网卡
环境变量，不需要手工填写 `eth0`、`enp*` 等接口名。

## 2. 启动双机服务

建议分别保留两个 SSH 窗口，以便持续查看两个节点的日志。

### master：10.44.53.212

进入 master 上的仓库根目录：

```bash
cd /实际路径/vllm_decode_profile
bash run_dp32/master.sh 2>&1 | tee master.log
```

启动命令后不必等待模型完成初始化，接着在另一台机器执行 headless 命令。

### headless：10.44.53.215

进入 headless 节点上的仓库根目录：

```bash
cd /实际路径/vllm_decode_profile
bash run_dp32/headless.sh 2>&1 | tee headless.log
```

headless 节点不提供 HTTP API；所有请求、profile 开关和健康检查都发往
`10.44.53.212:8077`。

master 启动时会打印类似信息：

```text
Starting node0 master on 10.44.53.212 (...)
Model: /mnt/models/GLM-5.1-w4a8
Profiling global DP rank 0, TP rank 0
Profile output: .../profiles/dp32_decode_...
```

记录 `Profile output:` 后面的目录，解析 profile 时会用到。

## 3. 检查服务是否就绪

等两台机器都完成模型加载后，在任意能访问 master 的终端执行：

```bash
curl -sS http://10.44.53.212:8077/v1/models
```

返回内容中应包含模型名 `glm-5`。然后可以发送一条普通在线推理请求：

```bash
curl -sS http://10.44.53.212:8077/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "glm-5",
    "messages": [
      {"role": "user", "content": "请简单介绍一下昇腾 NPU。"}
    ],
    "temperature": 0,
    "max_tokens": 64
  }'
```

## 4. 采集 decode profile

以下命令都只访问 master API。

### 4.1 启动 profile 窗口

```bash
curl -sS -X POST http://10.44.53.212:8077/start_profile
```

此时 DP0/TP0 只是进入等待状态；profiler 会跳过 prefill，直到遇到本地 BS=8 的
纯 decode step 才真正开始。

### 4.2 发送受控并发请求

默认目标是每个 DP rank 本地 BS=8。DP32 下可先使用约 256 个并发长请求，使
32 个 DP rank 平均各获得约 8 个请求。先准备请求体：

```bash
cat >/tmp/dp32_profile_request.json <<'JSON'
{
  "model": "glm-5",
  "messages": [
    {
      "role": "user",
      "content": "请从 1 开始逐项列出 200 个整数，每个整数单独占一行，不要省略。"
    }
  ],
  "temperature": 0,
  "max_tokens": 128
}
JSON
```

再并发发送 256 次：

```bash
seq 1 256 | xargs -P 256 -I{} \
  curl -sS -o /dev/null \
    http://10.44.53.212:8077/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -H 'X-Request-Id: dp32-profile-{}' \
    --data-binary @/tmp/dp32_profile_request.json
```

内部负载均衡不保证每次都恰好平均分配。如果 DP0 没有形成本地 BS=8，可根据
master 的 decode step 日志观察 DP0 的实际 `bsz`，然后调整并发量；也可以按
“常用覆盖项”一节把触发 batch 改为 `0`，让任意纯 decode batch 都能触发。

### 4.3 停止 profile 窗口

默认达到 8 个 decode step 后 profiler 会自动停止。无论是否已经自动停止，都
可以安全调用：

```bash
curl -sS -X POST http://10.44.53.212:8077/stop_profile
```

为了获得连续、稳定的 trace，采集窗口内尽量使用固定请求集合，不要持续注入
新的 prefill 请求。

## 5. 必查日志

master 上应该依次出现：

```text
VLLM_BASELINE_PROFILE_ARMED dp_rank=0 tp_rank=0 worker_rank=0 expected_local_batch=8 max_decode_steps=8
VLLM_BASELINE_PROFILE_STARTED dp_rank=0 tp_rank=0 worker_rank=0 local_batch=8 phase=decode
VLLM_BASELINE_PROFILE_STOPPED dp_rank=0 tp_rank=0 worker_rank=0 decode_steps=8 reason=max_decode_steps
```

decode step 会批量打印，例如：

```text
VLLM_DECODE_STEP_LOG_BEGIN dp_rank=0 count=2 reason=step_limit
[VLLM_DECODE step=0001 dp_rank=0] bsz=8, num_tokens=8, TPOT=68.420 ms, seq_lens=[...], HBM_KV=1407/1680 blocks (83.75%)
[VLLM_DECODE step=0002 dp_rank=0] bsz=8, num_tokens=8, TPOT=67.981 ms, seq_lens=[...], HBM_KV=1407/1680 blocks (83.75%)
VLLM_DECODE_STEP_LOG_END
```

这些都是 DP0 EngineCore 的本地数据：

- `bsz`、`num_tokens` 和 `seq_lens` 不是 DP32 的全局汇总值；
- `HBM_KV` 是目标 NPU 的本地 KV block 占用；
- `TPOT` 是该 EngineCore 从开始调度到处理完模型输出的 step 墙钟时间。

另外 31 个 DP rank 不应打印 profile marker，也不应创建 profiler。

## 6. 解析 profile

在 master 上进入仓库根目录，把下面路径替换为启动日志中 `Profile output:`
打印的实际目录：

```bash
python3 parse_profile.py /实际路径/vllm_decode_profile/profiles/dp32_decode_时间戳
```

脚本会找到唯一的 `*ascend_pt` 原始目录，调用 Ascend profiler analyse，并打印
最终的 `trace_view.json` 路径。可以使用 MindStudio Insight 打开该文件。

因为默认只采一张卡，所以目标目录中应该只有一个 `*ascend_pt`。

## 7. 常用覆盖项

脚本中的默认值都可以在启动命令前用环境变量覆盖：

| 环境变量 | master 默认值 | headless 默认值 | 说明 |
| --- | --- | --- | --- |
| `VLLM_NODE_LOCAL_IP` | `10.44.53.212` | `10.44.53.215` | 当前节点用于通信的 IP |
| `VLLM_NODE0_IP` | `10.44.53.212` | `10.44.53.212` | DP coordinator 地址 |
| `VLLM_MODEL` | `/mnt/models/GLM-5.1-w4a8` | 同左 | 本地模型目录 |
| `ASCEND_RT_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15` | 同左 | 当前节点使用的 16 张 NPU |
| `VLLM_PROFILE_TARGET_DP_RANK` | `0` | `0` | 唯一采集 profile 的全局 DP rank |
| `VLLM_PROFILE_TARGET_TP_RANK` | `0` | `0` | 目标 DP EngineCore 内的 TP rank |
| `VLLM_PROFILE_TRIGGER_BATCH_SIZE` | `8` | `8` | 触发采集的本地纯 decode batch；`0` 表示任意 |
| `VLLM_PROFILE_MAX_DECODE_STEPS` | `8` | `8` | 最多采集的连续 decode step；`0` 表示不限 |
| `VLLM_PROFILE_STOP_ON_PHASE_CHANGE` | `1` | `1` | decode 阶段改变时自动停止 |
| `VLLM_DECODE_LOG_DP_RANK` | `0` | `0` | 唯一打印 decode 统计的 Scheduler |
| `VLLM_DECODE_LOG_FLUSH_STEPS` | `16` | `16` | 累计多少条 decode 日志后批量打印 |
| `VLLM_DECODE_LOG_FLUSH_SECONDS` | `0` | `0` | 按秒刷新；`0` 表示禁用 |
| `VLLM_PROFILE_DIR` | 仓库下的时间戳目录 | 同左 | profiler 输出目录，启动时必须为空 |

例如，只想先验证任意本地 batch 都能采到，并只采 4 步，可以在两台机器启动前
都执行：

```bash
export VLLM_PROFILE_TRIGGER_BATCH_SIZE=0
export VLLM_PROFILE_MAX_DECODE_STEPS=4
```

然后分别运行原来的 `master.sh` 和 `headless.sh`。

## 8. 常见问题

### `no network interface found for local IP`

脚本没有在本机发现预设 IP。执行 `ip -o -4 addr show` 核对地址；如果机器 IP
发生变化，用 `VLLM_NODE_LOCAL_IP` 覆盖当前节点 IP，并确保
`VLLM_NODE0_IP` 始终指向 master。

### `VLLM_PROFILE_DIR must be empty`

该目录中存在旧 profile。换一个新的 `VLLM_PROFILE_DIR`，不要把多次采集写入
同一目录。

### 出现 `VLLM_BASELINE_PROFILE_NOT_STARTED`

说明 DP0 在本次窗口中没有形成符合条件的纯 decode step。优先检查：

- master 日志中 DP0 的实际 `bsz`；
- 全局并发请求数量和每个请求的生成 token 数；
- 是否不断有新 prefill 混入；
- KV Cache 是否充足；
- `VLLM_PROFILE_TRIGGER_BATCH_SIZE` 是否设得过于严格。

### 两个节点一直等待或初始化失败

先确认两台机器使用相同的模型权重和兼容的软件版本，再检查：

```bash
ip -o -4 addr show
npu-smi info
vllm --version
python3 -c 'import vllm, vllm_ascend, torch_npu; print("imports OK")'
```

同时保留 `master.log` 和 `headless.log`。如果仍然失败，请提供两台机器上这些
命令的输出，以及两个日志中第一处 traceback 或 error 前后的内容。
