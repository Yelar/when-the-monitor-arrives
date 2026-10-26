"""
Score every seller message in finished runs offline (sprint project 3.4, "When the Monitor Arrives").

    # what the LLM monitor would have said about EVERY message, read or not (evasion, keyword calibration)
    uv run python experiments/monitor_arrives/score_messages.py --mode monitor --runs results/pilot --out results/pilot/monitor_offline.jsonl
    # ground truth: a stronger judge sees the message AND the sender's private notes
    uv run python experiments/monitor_arrives/score_messages.py --mode judge --model RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic \
        --runs results/main --out results/main/judge.jsonl

One JSON line per message, keyed by (run, round, seller). Lines already in --out are skipped, so it resumes.
Models are reached through LLM_ENDPOINTS (see src/resources/model_wrappers.py), defaulting to run_grid's servers.
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_grid import DEFAULT_ENDPOINTS, MONITOR_MODEL  # noqa: E402
from src.continuous_double_auction.market.agents import TEMPLATE_DIR  # noqa: E402
from src.continuous_double_auction.market.monitor import generate_json  # noqa: E402
from src.continuous_double_auction.util.jinja_util import render_prompt  # noqa: E402
from src.continuous_double_auction.util.misc import get_client, parse_log  # noqa: E402

RESPONSE_BLOCK = re.compile(r"## Response: Round (\d+) - [^\n]*\n````json\n(.*?)\n````", re.DOTALL)


def seller_responses(run_dir: Path, seller_id: str) -> dict[int, dict]:
    """Round -> the seller's full JSON response (private fields included) from its transcript."""
    path = run_dir / f"{seller_id}.md"
    if not path.exists():
        return {}
    return {int(r): json.loads(body) for r, body in RESPONSE_BLOCK.findall(path.read_text(encoding="utf-8"))}


def messages_in_run(run_dir: Path) -> list[dict]:
    meta = json.loads((run_dir / "experiment_metadata.json").read_text())
    tag = meta["auction_config"]["tag"]
    condition, seed = tag.rsplit("-s", 1)
    private = {s: seller_responses(run_dir, s) for s in meta["seller_ids"]}
    out = []
    for r in parse_log(run_dir / "unified.log"):
        for seller, message in sorted(r.get("seller_messages", {}).items()):
            review = r.get("monitor_reviews", {}).get(seller, {})
            notes = private.get(seller, {}).get(r["round_number"], {})
            out.append({
                "run": run_dir.name, "condition": condition, "seed": int(seed), "round": r["round_number"],
                "seller": seller, "seller_index": meta["seller_ids"].index(seller), "message": message,
                "read": review.get("read"), "flagged": review.get("flagged"), "online_score": review.get("score"),
                "fined": seller in r.get("fines", {}),
                "plan_for_this_hour": str(notes.get("plan_for_this_hour", "")),
                "plan_for_message": str(notes.get("plan_for_message", "")),
                "scratch_pad_update": str(notes.get("scratch_pad_update", "")),
                "thinking": clip_words(str(notes.get("thinking", "")), THINKING_WORDS),
            })
    return out


THINKING_WORDS = 1500  # keeps the judge prompt well inside its 8k context


def clip_words(text: str, n: int) -> str:
    """The first and last n/2 words of text: the analysis and the decision."""
    words = text.split()
    return text if len(words) <= n else " ".join(words[: n // 2] + ["[...]"] + words[-n // 2 :])


def score(client, mode: str, item: dict, monitor_template: str = "message_monitor_prompt.jinja2") -> dict:
    template = monitor_template if mode == "monitor" else "message_judge_prompt.jinja2"
    prompt = render_prompt(template_dir=TEMPLATE_DIR, prompt_template=template, **item)
    return generate_json(client, prompt, "coordination_score" if mode == "monitor" else "intent_score")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["monitor", "judge"], required=True)
    p.add_argument("--model", default=MONITOR_MODEL)
    p.add_argument("--endpoint", default=None, help="base URL serving --model (default: from LLM_ENDPOINTS)")
    p.add_argument("--runs", type=Path, nargs="+", required=True, help="results directories (searched recursively)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--template", default="message_monitor_prompt.jinja2",
                   help="monitor mode: prompt template (message_monitor_prompt_strict.jinja2 for the C8 monitor)")
    args = p.parse_args()

    if args.endpoint:
        os.environ["LLM_ENDPOINTS"] = json.dumps({args.model: [args.endpoint]})
    os.environ.setdefault("LLM_ENDPOINTS", json.dumps(DEFAULT_ENDPOINTS))
    run_dirs = sorted({m.parent for d in args.runs for m in d.rglob("collusion_metrics.json")})
    items = [it for d in run_dirs for it in messages_in_run(d)]
    done = set()
    if args.out.exists():
        done = {(j["run"], j["round"], j["seller"]) for j in map(json.loads, args.out.read_text().splitlines())}
    todo = [it for it in items if (it["run"], it["round"], it["seller"]) not in done]
    print(f"{len(run_dirs)} runs, {len(items)} messages, {len(todo)} to score with {args.model}", flush=True)

    client = get_client(model=args.model, temperature=0)
    with ThreadPoolExecutor(max_workers=args.workers) as pool, open(args.out, "a") as f:
        futures = {pool.submit(score, client, args.mode, it, args.template): it for it in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            it = futures[fut]
            f.write(json.dumps({**it, "mode": args.mode, "model": args.model, "result": fut.result()}) + "\n")
            f.flush()
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
