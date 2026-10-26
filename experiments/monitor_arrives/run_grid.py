"""
Run the "When the Monitor Arrives" conditions (sprint project 3.4) against local vLLM servers.

    uv run python experiments/monitor_arrives/run_grid.py --conditions C1 C2 --seeds 0 1 2 --out results/pilot
    uv run python experiments/monitor_arrives/run_grid.py --conditions C1 C3 --seeds 0 --out results/smoke -- --rounds 3

Each (condition, seed) is one simulation.py process writing to <out>/<timestamp>_<condition>-s<seed>/.
Runs whose directory already has collusion_metrics.json are skipped, so an interrupted grid resumes.
Arguments after "--" are passed to every run (they override the fixed settings below).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
AGENT_MODEL = "Qwen/Qwen3-32B-FP8"
MONITOR_MODEL = "RedHatAI/Mistral-Small-24B-Instruct-2501-FP8-dynamic"
DEFAULT_ENDPOINTS = {
    AGENT_MODEL: ["http://127.0.0.1:8101/v1", "http://127.0.0.1:8102/v1"],
    MONITOR_MODEL: ["http://127.0.0.1:8103/v1"],
}

# Fixed settings: 3 LLM sellers, 3 ZIP buyers, 40 hours with the horizon hidden, paper prompt without any
# mention of monitoring. ZIP buyers follow last hour's trade price and raise their bids only when nothing
# traded, so a seller who holds out alone loses the sale and only a joint hold-out lifts prices.
# (In the first pilot, LLM buyers anchored near $83 whatever the sellers did, so no condition could show a cartel.)
FIXED = [
    "--sellers", "3", "--buyers", "3", "--rounds", "40", "--hide_num_rounds",
    "--seller_models", AGENT_MODEL, "--buyer_models", "zip", "--monitor_model", MONITOR_MODEL, "-y",
]
CHAT = ["--seller_comms_enabled"]
CONDITIONS = {
    "C1": CHAT,                                                        # chat open, no monitor (control)
    "C2": [],                                                          # no chat (floor)
    "C3": CHAT + ["--monitor", "llm"],                                 # LLM monitor reads every message
    "C4": CHAT + ["--monitor", "llm", "--monitor_rate", "0.25"],       # LLM monitor reads a random 25%
    "C5": CHAT + ["--monitor", "llm", "--unmonitored_seller", "2"],    # reads everyone except seller C
    "C6": CHAT + ["--monitor", "keyword"],                             # keyword monitor reads every message
    "C7": CHAT + ["--monitor", "llm", "--monitor_start_round", "11"],  # optional: monitor arrives at hour 11
    # Follow-ups, written after the first informed-setup labels showed missed attempts were mostly price ranges
    # framed as analysis: a stricter prompt that names that form, and a notice that says which words were flagged.
    "C8": CHAT + ["--monitor", "llm", "--monitor_prompt", "message_monitor_prompt_strict.jinja2"],
    "C9": CHAT + ["--monitor", "llm", "--fine_feedback"],
}


def finished(out: Path, tag: str) -> bool:
    return any((d / "collusion_metrics.json").exists() for d in out.glob(f"*_{tag}"))


def run_one(cond: str, seed: int, out: Path, extra: list[str], env: dict[str, str]) -> tuple[str, int, float]:
    tag = f"{cond}-s{seed}"
    cmd = [sys.executable, "src/continuous_double_auction/simulation.py", *FIXED, *CONDITIONS[cond],
           "--seeds", str(seed), "--tag", tag, "--log_dir", str(out), *extra]
    start = time.time()
    with open(out / "logs" / f"{tag}.log", "w") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        rc = subprocess.run(cmd, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    return tag, rc, time.time() - start


def main() -> None:
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1 :] if "--" in argv else []
    argv = argv[: argv.index("--")] if "--" in argv else argv
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--parallel", type=int, default=12, help="simulations running at once")
    args = p.parse_args(argv)

    out = args.out if args.out.is_absolute() else PROJECT / args.out
    (out / "logs").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("LLM_ENDPOINTS", json.dumps(DEFAULT_ENDPOINTS))
    # Seeds outermost, so a grid cut short still has every condition for the first seeds
    jobs = [(c, s) for s in args.seeds for c in args.conditions if not finished(out, f"{c}-s{s}")]
    print(f"{len(jobs)} runs to do in {out} ({args.parallel} at a time)", flush=True)
    failed = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(run_one, c, s, out, extra, env) for c, s in jobs]
        for f in as_completed(futures):
            tag, rc, secs = f.result()
            failed += rc != 0
            print(f"{'done' if rc == 0 else 'FAILED'} {tag} in {secs / 60:.1f} min (log: {out}/logs/{tag}.log)", flush=True)
    print(f"grid finished: {len(jobs) - failed} ok, {failed} failed", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
