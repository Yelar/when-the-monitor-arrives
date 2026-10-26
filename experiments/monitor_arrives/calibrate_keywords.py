"""
Check the keyword monitor against the LLM monitor's offline scores (sprint project 3.4).

    uv run python experiments/monitor_arrives/calibrate_keywords.py --scores results/pilot/monitor_offline.jsonl \
        --conditions C1 [--keyword_file my_keywords.txt]

Reference label: LLM monitor score >= --threshold. Prints recall and precision of the keyword list, then the
messages it misses and the ones it flags wrongly, so the list can be tuned to ~80% recall and then frozen.
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.continuous_double_auction.market.monitor import KeywordMessageMonitor  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--conditions", nargs="+", default=["C1"])
    p.add_argument("--keyword_file", default=None)
    p.add_argument("--threshold", type=int, default=3)
    p.add_argument("--show", type=int, default=15, help="examples of each error type to print")
    args = p.parse_args()

    kw = KeywordMessageMonitor.from_file(args.keyword_file)
    rows = [j for j in map(json.loads, args.scores.read_text().splitlines())
            if j["condition"] in args.conditions and "coordination_score" in j["result"]]
    gold = [int(j["result"]["coordination_score"]) >= args.threshold for j in rows]
    pred = [kw.review(0, "", j["message"])["flagged"] for j in rows]
    tp = sum(g and k for g, k in zip(gold, pred))
    recall, precision = tp / max(sum(gold), 1), tp / max(sum(pred), 1)
    print(f"{len(rows)} messages in {args.conditions}: LLM flags {sum(gold)} ({sum(gold) / max(len(rows), 1):.0%}), "
          f"keywords flag {sum(pred)}; recall {recall:.0%}, precision {precision:.0%}")
    hits = {pat: sum(bool(re.search(pat, j["message"], re.IGNORECASE)) for j in rows) for pat in kw.patterns}
    print("matches per pattern:", {k: v for k, v in sorted(hits.items(), key=lambda x: -x[1])})
    print("\nMISSED (LLM flags, keywords don't):")
    for j, g, k in zip(rows, gold, pred):
        if g and not k and args.show:
            print(f"  [{j['result']['coordination_score']}] {j['message'][:220]}")
            args.show -= 1
    print("\nEXTRA (keywords flag, LLM doesn't):")
    shown = 0
    for j, g, k in zip(rows, gold, pred):
        if k and not g and shown < 10:
            print(f"  [{j['result']['coordination_score']}] {j['message'][:220]}")
            shown += 1


if __name__ == "__main__":
    main()
