# When the Monitor Arrives

Code for project 3.4 of the Apart Research AI Collusion Research Sprint (October 2026), by Yelarys Yertaiuly (MBZUAI).
It studies what LLM sellers in a simulated double-auction market do when an unannounced LLM monitor fines coordination
in their messages: do they stop coordinating, or reword their messages so the monitor misses them?

- Experiments, labeling and analysis, with run instructions: [`experiments/monitor_arrives/README.md`](experiments/monitor_arrives/README.md)
- The monitor: `src/continuous_double_auction/market/monitor.py`, with its prompts in `src/continuous_double_auction/prompt_templates/`
- Tests: `tests/test_monitor.py` and `tests/test_zip_buyer.py`

## Preexisting code

The market simulator in `src/continuous_double_auction/` is preexisting code from Agrawal et al. (2025), *Evaluating
LLM Agent Collusion in Double Auctions* ([arXiv:2507.01413](https://arxiv.org/abs/2507.01413)), as adapted in the
sprint's repository ([poseidon-research/collusion_hackathon](https://github.com/poseidon-research/collusion_hackathon)).
The monitor, the simulator changes listed in `experiments/monitor_arrives/README.md`, the two tests and everything in
`experiments/monitor_arrives/` were written during the sprint. Run outputs (about 570 MB of transcripts, scores and
labels) are not included.
