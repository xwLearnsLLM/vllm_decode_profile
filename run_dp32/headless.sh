#!/usr/bin/env bash

set -euo pipefail

# Node 1: headless worker hosting global DP ranks 16-31.
local_ip="${VLLM_NODE_LOCAL_IP:-10.44.53.215}"
node0_ip="${VLLM_NODE0_IP:-10.44.53.212}"
model_path="${VLLM_MODEL:-/mnt/models/GLM-5.1-w4a8}"
nic_name="${VLLM_NIC_NAME:-}"

if [ -z "$nic_name" ] && command -v ip >/dev/null 2>&1; then
  nic_name="$(
    ip -o -4 addr show |
      awk -v ip="$local_ip" '
        {
          split($4, addr, "/")
          if (addr[1] == ip) {
            name = $2
            sub(/@.*/, "", name)
            print name
            exit
          }
        }
      ' || true
  )"
fi

if [ -z "$nic_name" ] && command -v python3 >/dev/null 2>&1; then
  nic_name="$(
    python3 - "$local_ip" 2>/dev/null <<'PY' || true
import fcntl
import os
import socket
import struct
import sys

target_ip = sys.argv[1]
for interface in sorted(os.listdir("/sys/class/net")):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        request = struct.pack("256s", interface[:15].encode())
        response = fcntl.ioctl(sock.fileno(), 0x8915, request)
        interface_ip = socket.inet_ntoa(response[20:24])
    except OSError:
        continue
    finally:
        sock.close()
    if interface_ip == target_ip:
        print(interface)
        break
PY
  )"
fi

if [ -z "$nic_name" ]; then
  echo "Error: no network interface found for local IP $local_ip" >&2
  echo "Set it explicitly, for example: VLLM_NIC_NAME=eth0 bash run_dp32/headless.sh" >&2
  exit 1
fi

if [ ! -d "/sys/class/net/$nic_name" ]; then
  echo "Error: network interface does not exist: $nic_name" >&2
  exit 1
fi

if [ ! -d "$model_path" ]; then
  echo "Error: model directory does not exist: $model_path" >&2
  exit 1
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

profile_run_id="${VLLM_PROFILE_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export VLLM_PROFILE_DIR="${VLLM_PROFILE_DIR:-$repo_root/profiles/dp32_decode_$profile_run_id}"

# These selectors deliberately point at node 0. Node 1 receives the broadcast
# profile commands, but every local worker returns without creating a profiler.
export VLLM_PROFILE_TARGET_DP_RANK="${VLLM_PROFILE_TARGET_DP_RANK:-0}"
export VLLM_PROFILE_TARGET_TP_RANK="${VLLM_PROFILE_TARGET_TP_RANK:-0}"
export VLLM_PROFILE_TRIGGER_BATCH_SIZE="${VLLM_PROFILE_TRIGGER_BATCH_SIZE:-8}"
export VLLM_PROFILE_MAX_DECODE_STEPS="${VLLM_PROFILE_MAX_DECODE_STEPS:-8}"
export VLLM_PROFILE_STOP_ON_PHASE_CHANGE="${VLLM_PROFILE_STOP_ON_PHASE_CHANGE:-1}"
export VLLM_DECODE_LOG_DP_RANK="${VLLM_DECODE_LOG_DP_RANK:-0}"
export VLLM_DECODE_LOG_FLUSH_STEPS="${VLLM_DECODE_LOG_FLUSH_STEPS:-16}"
export VLLM_DECODE_LOG_FLUSH_SECONDS="${VLLM_DECODE_LOG_FLUSH_SECONDS:-0}"

if [ -d "$VLLM_PROFILE_DIR" ] &&
  [ -n "$(find "$VLLM_PROFILE_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
  echo "Error: VLLM_PROFILE_DIR must be empty: $VLLM_PROFILE_DIR" >&2
  exit 1
fi
mkdir -p "$VLLM_PROFILE_DIR"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP="$local_ip"
export GLOO_SOCKET_IFNAME="$nic_name"
export TP_SOCKET_IFNAME="$nic_name"
export HCCL_SOCKET_IFNAME="$nic_name"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export VLLM_ASCEND_BALANCE_SCHEDULING=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export PYTHONHASHSEED=114514
export VLLM_ASCEND_ENABLE_MLAPO=1

echo "Starting node1 headless worker on $local_ip ($nic_name)"
echo "Model: $model_path"
echo "Profile target remains global DP rank $VLLM_PROFILE_TARGET_DP_RANK on node0"

exec vllm serve "$model_path" \
  --host 0.0.0.0 \
  --port 8077 \
  --headless \
  --data-parallel-size 32 \
  --data-parallel-size-local 16 \
  --data-parallel-start-rank 16 \
  --data-parallel-address "$node0_ip" \
  --data-parallel-rpc-port 12890 \
  --tensor-parallel-size 1 \
  --seed 1024 \
  --no-enable-prefix-caching \
  --served-model-name glm-5 \
  --enable-expert-parallel \
  --all2all-backend flashinfer_all2allv \
  --max-num-seqs 16 \
  --max-model-len 20480 \
  --max-num-batched-tokens 16384 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --enable-chunked-prefill \
  --block-size 128 \
  --worker-cls vllm_decode_profile.profile_worker.DecodeOnlyRankFilteredNPUWorker \
  --scheduler-cls vllm_decode_profile.profile_scheduler.DecodeStepLoggingScheduler \
  --profiler-config.profiler=torch \
  --profiler-config.torch_profiler_dir="$VLLM_PROFILE_DIR" \
  --profiler-config.torch_profiler_with_stack=false \
  --profiler-config.torch_profiler_record_shapes=false \
  --profiler-config.torch_profiler_with_memory=false \
  --profiler-config.ignore_frontend=true \
  --additional-config '{"dsa_sparse_config": {"enabled": true, "split_indexer_cache": true, "hbm_sparse_budget": 2048, "indexer_mla_block_ratio": 3, "max_active_reqs": 256, "hot_cpu_block_multiple": 1, "trace_points": {"enabled": false, "points": ["lightning_indexer", "gather_selection"], "ranks": [0], "layers": [0], "sync": false}, "enable_row_mode_decode_graph": true}}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8], "mode": "VLLM_COMPILE"}'
