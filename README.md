
# Topological-MCTS-ARC

A clean, reproducible scaffold for **Topological MCTS** on ARC-style grid completion tasks.


## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Run a tiny baseline (TMCTS only, 10 tasks):

```bash
python experiments/run_baselines.py --trials 1 --iterations 100
```

Results land in `experiments/results/baselines.csv`.

## Layout
- `src/topomcts` — package modules (`game.py`, `complex.py`, `invariants.py`, `mcts.py`, `meta.py`, `datasets.py`, `metrics.py`)
- `experiments/` — scripts to reproduce tables/figures; CSV outputs in `experiments/results`
- `notebooks/legacy.ipynb` — auto-copied from the uploaded repo (if available)
- `src/topomcts/legacy_port.py` — concatenated code cells from `legacy.ipynb` for quick reference
- `tests/` — smoke tests for core pieces

## Next steps
- Implement ablations in `experiments/run_ablation.py`
- Add signature distance & transfer logic in `experiments/run_transfer.py`
- Add curriculum study + simple plots in `experiments/run_curriculum.py`
- Replace placeholder symmetry/invariants with your article's precise definitions
- Expand `datasets.py` to match the exact dataset spec (counts, sizes, seeds) used in the paper
