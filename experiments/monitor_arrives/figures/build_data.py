"""
Collect everything the report's figures and tables need into one file, figures/data.json.

    uv run python experiments/monitor_arrives/figures/build_data.py

Reads, for each setup, the analysis summary (analysis/summary.json), the offline monitor and judge scores, and the
blind labels in llm_labels/ (key.json maps L01.. to message ids; labels_<labeler><batch>.json hold the labels), and
the summary of the Claude labels of every message (analysis/claude_labels_summary.json).
"""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
SETUPS = {"base": "results/main", "informed": "results/main_informed"}


def clean(x):
    """NaN is not valid JSON; write it as null."""
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean(v) for v in x]
    return x


def message_key(j: dict) -> str:
    return f"{j['run']}|{j['round']}|{j['seller']}"


def label_sample(d: Path, judge: dict, monitor: dict) -> list[dict]:
    lab = d / "llm_labels"
    key = json.loads((lab / "key.json").read_text())
    labels = {who: {} for who in "AB"}
    for f in sorted(lab.glob("labels_*.json")):
        labels[f.stem.split("_")[1][0]].update(json.loads(f.read_text()))
    rows = {r["id"]: r for r in csv.DictReader(open(d / "hand_labels.csv"))}
    out = []
    for lid, mid in key.items():
        j, m = judge[mid], monitor[mid]
        out.append({
            "label_id": lid, "condition": rows[mid]["condition"], "hour": int(rows[mid]["hour"]),
            "message": rows[mid]["message"],
            "labeler_a": labels["A"][lid]["intent"], "labeler_b": labels["B"].get(lid, {}).get("intent"),
            "concealment_a": labels["A"][lid]["concealment"],
            "concealment_b": labels["B"].get(lid, {}).get("concealment"),
            "mentions_monitoring_a": labels["A"][lid]["mentions_monitoring"],
            "mentions_monitoring_b": labels["B"].get(lid, {}).get("mentions_monitoring"),
            "judge": j["result"].get("intent_score"), "judge_mentions_monitoring": j["result"].get("mentions_monitoring"),
            "monitor": m["result"].get("coordination_score"), "fined": bool(m.get("fined")),
        })
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=HERE / "data.json")
    args = p.parse_args()
    data = {}
    for setup, rel in SETUPS.items():
        d = PROJECT / rel
        summary = json.loads((d / "analysis" / "summary.json").read_text())
        judge = {message_key(j): j for j in map(json.loads, open(d / "judge.jsonl"))}
        monitor = {message_key(j): j for j in map(json.loads, open(d / "monitor_offline.jsonl"))}
        fines = Counter((j["condition"], j["run"], j["seller"]) for j in monitor.values() if j.get("fined"))
        per_seller = {}
        for (cond, _, _), n in fines.items():
            per_seller.setdefault(cond, []).append(n)
        data[setup] = {
            "conditions": summary["conditions"],
            "coordination_rates": summary["coordination_rates"],
            "late_monitor_did": summary["c7_did"],
            "discovery": summary["discovery"],
            "signatures": summary["signatures"],
            "judge_summary": summary["judge"],
            "runs": [{k: r[k] for k in ("dir", "condition", "seed", "trade_index")} for r in summary["runs"]],
            "fines_per_fined_seller": {c: {"max": max(v), "mean": sum(v) / len(v), "sellers": len(v)}
                                       for c, v in sorted(per_seller.items())},
            "label_sample": label_sample(d, judge, monitor),
            # every message labelled by Claude (claude_labels.py), summarized by claude_label_stats.py
            "claude_labels": json.loads((d / "analysis" / "claude_labels_summary.json").read_text()),
        }
    rob = PROJECT / "results" / "robustness.json"  # robustness.py: hourly series and the follow-up checks
    if rob.exists():
        data["robustness"] = json.loads(rob.read_text())
    args.out.write_text(json.dumps(clean(data), indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
