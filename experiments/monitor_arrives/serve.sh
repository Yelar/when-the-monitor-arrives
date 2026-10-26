#!/usr/bin/env bash
# Start the local vLLM servers for "When the Monitor Arrives" (sprint project 3.4), one tmux window each.
#   GPUs 4,5 -> agent model copy A (port 8101)   GPUs 6,7 -> agent model copy B (port 8102)
#   GPU 2    -> monitor model       (port 8103)
#   `serve.sh judge`: GPUs 4-7 -> offline judge (port 8104), after the runs, once agentA/agentB are stopped
# Refuses to start on a GPU that already has memory in use, so it never lands on someone else's job.
# Servers listen on 127.0.0.1 only. Stop them with:  tmux kill-window -t cm:agentA  (etc.)
set -euo pipefail

ROOT=${ROOT:-$HOME/projects/collusion-monitor}
VLLM=${VLLM:-$ROOT/.venv-vllm/bin/vllm}
LOGS=${LOGS:-$ROOT/logs}
SESSION=${SESSION:-cm}
AGENT_MODEL=${AGENT_MODEL:-Qwen/Qwen3-32B-FP8}
MONITOR_MODEL=${MONITOR_MODEL:-RedHatAI/Mistral-Small-24B-Instruct-2501-FP8-dynamic}
mkdir -p "$LOGS"
tmux has-session -t "$SESSION" 2>/dev/null || tmux new-session -d -s "$SESSION" -n shell

gpus_free() {  # every listed GPU has < 200 MiB in use
  local used
  for g in ${1//,/ }; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    if (( used > 200 )); then echo "GPU $g already has ${used} MiB in use; not starting $2" >&2; return 1; fi
  done
}

serve() {  # name gpus port model [vllm args...]
  local name=$1 gpus=$2 port=$3 model=$4 cmd; shift 4
  gpus_free "$gpus" "$name" || return 0
  # %q keeps JSON arguments intact through tmux's shell; PCI order makes CUDA ids match nvidia-smi.
  # No NVLink here and NCCL's PCIe peer-to-peer setup hangs, so tensor-parallel traffic goes through host memory.
  printf -v cmd '%q ' "$VLLM" serve "$model" --host 127.0.0.1 --port "$port" "$@"
  tmux new-window -d -t "$SESSION" -n "$name" \
    "CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpus NCCL_P2P_DISABLE=1 $cmd 2>&1 | tee $LOGS/$name.log"
  echo "started $name on GPU $gpus, port $port"
}

which=${1:-all}
if [[ $which == all || $which == agentA ]]; then
  serve agentA 4,5 8101 "$AGENT_MODEL" --tensor-parallel-size 2 --max-model-len 16384 \
    --gpu-memory-utilization 0.90 --max-num-seqs 48 --disable-custom-all-reduce
fi
if [[ $which == all || $which == agentB ]]; then
  serve agentB 6,7 8102 "$AGENT_MODEL" --tensor-parallel-size 2 --max-model-len 16384 \
    --gpu-memory-utilization 0.90 --max-num-seqs 48 --disable-custom-all-reduce
fi
if [[ $which == judge ]]; then
  # Offline judge only, after the runs: needs GPUs 4-7, so stop agentA and agentB first
  serve judge 4,5,6,7 8104 "${JUDGE_MODEL:-RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic}" --tensor-parallel-size 4 \
    --max-model-len 8192 --gpu-memory-utilization 0.90 --max-num-seqs 64 --disable-custom-all-reduce
fi
if [[ $which == all || $which == monitor ]]; then
  # Text-only Mistral Small 3 (2501). Vision-language checkpoints (Mistral-Small-3.2, Gemma-3, even unquantized
  # Gemma-3-12B) all produced garbage under this vLLM 0.11 + transformers 4.57 install, while text-only models work.
  serve monitor 2 8103 "$MONITOR_MODEL" --max-model-len 8192 --gpu-memory-utilization 0.92 --max-num-seqs 32
fi
