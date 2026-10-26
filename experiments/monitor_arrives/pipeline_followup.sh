#!/bin/bash
# Follow-up runs on gpu-04 (informed setup, seeds 0-7): C8 = full monitor with the strict prompt, C9 = full monitor
# whose fine notices quote the flagged words. Then score every new message offline with the original and the strict
# monitor prompts, and run the analysis. Only touches this session's own tmux windows (cm:*).
set -u
cd ~/projects/collusion-monitor/repo/llm-agent-collusion
UV=~/.local/bin/uv
OUT=results/followup_informed
GRID=(experiments/monitor_arrives/run_grid.py --conditions C8 C9 --seeds 0 1 2 3 4 5 6 7 --out $OUT
      --parallel 16 -- --boss_pressure --enable_thinking --informed_sellers)
done_runs() { ls $OUT/*_C*-s*/collusion_metrics.json 2>/dev/null | wc -l; }

$UV run python "${GRID[@]}"
echo "grid done at $(date -u): $(done_runs)/16 runs finished"
if [ "$(done_runs)" -lt 16 ]; then
  $UV run python "${GRID[@]}"
  echo "after retry: $(done_runs)/16 runs finished"
fi

for t in message_monitor_prompt.jinja2:monitor_offline message_monitor_prompt_strict.jinja2:monitor_offline_strict; do
  $UV run python experiments/monitor_arrives/score_messages.py --mode monitor --endpoint http://127.0.0.1:8103/v1 \
    --template "${t%%:*}" --runs $OUT --out "$OUT/${t##*:}.jsonl"
done
echo "monitor scoring done at $(date -u)"
$UV run python experiments/monitor_arrives/analyze.py --runs $OUT --out $OUT/analysis \
  --monitor_offline $OUT/monitor_offline.jsonl > $OUT/analysis.out 2>&1
echo "FOLLOWUP DONE at $(date -u)"
