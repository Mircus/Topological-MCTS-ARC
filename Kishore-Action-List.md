
# Kishore — Action List (TopologicalMCTS‑ARC)
**Deadline:** Oct 15  
**Goal:** Make the paper’s experiments fully reproducible from this repo: `pip install -e .` → run scripts → CSVs + figures that match the article’s tables.

---

## P0 — Must‑have for reproducibility
- [ ] **Repo sanity + packaging**
  - [ ] Verify `pip install -e .` works in a fresh venv on macOS/Linux/Windows.
  - [ ] Keep only `src/topomcts/*` as the import surface; legacy code stays in `src/topomcts/legacy_port.py` for reference.
- [ ] **Dataset spec (single source of truth) — `src/topomcts/datasets.py`**
  - [ ] Implement the exact paper spec (counts, grid‑mix, seeds):
    - Grid mix: 3×3 (60%), 4×4 (25%), 5×5 (15%)
    - Alphabet `{0..4}`
    - Missing cells ∈ {1,2,3} (mean ~1.7)
    - Depth bound policy: `dmax = 2*(#missing)+1`
  - [ ] Emit a **manifest JSON** with per‑task metadata (`id, grid, missing, seed, dmax`).
  - [ ] Preserve **train/val/test splits** and **global seeds**.
- [ ] **Topological bonus API — `src/topomcts/mcts.py`**
  - [ ] Refactor `TopologicalMCTSEngine.topological_bonus(state)` into a **composable mix**: `bonus = w_c*centrality + w_d*diffusion + w_s*symmetry`.
  - [ ] Expose weights via CLI/Config. Default placeholder: `w_c=0.4, w_d=0.3, w_s=0.3` (update to paper values if different).
- [ ] **Logging & summaries**
  - [ ] For every run, write **CSV** with: `task_id, method, success, quality, sims, time_s, topo_c, topo_d, topo_s, dmax, seed`.
  - [ ] Aggregate to a **JSON summary** (means, std, N) to feed figures/tables.
- [ ] **`experiments/run_baselines.py` — Table 2/3**
  - [ ] Implement the following methods (flags to toggle):
    - `mcts_std` (vanilla UCB1)
    - `mcts_heur` (vanilla + simple domain heuristic)
    - `mcts_neural` (optional stub: MLP policy/value)
    - `ph_features` (optional: Gudhi/Giotto‑TDA)
    - `tmcts` (our method: UCB1 + topo bonus)
    - `gnn_transfer` (optional stub for comparison)
  - [ ] Output: `experiments/results/baselines.csv`

---

## P1 — Ablations, Transfer, Curriculum
- [ ] **Ablations — `experiments/run_ablation.py` (Table 4)**
  - [ ] Implement variants:
    - `fiedler_only` (centrality only)
    - `diffusion_only`
    - `symmetry_only`
    - `all_features` (full mix)
  - [ ] Produce `experiments/results/ablation.csv` and a bar chart in `experiments/figures/ablation.png`.
- [ ] **Transfer — `experiments/run_transfer.py`**
  - [ ] In `src/topomcts/invariants.py`, finalize signature `I(G) = (β0, β1, symmetry, spectral_radius, …)`; keep it **lightweight & stable**.
  - [ ] Define `d_topo(I_i, I_j)` (normalized L2 or Mahalanobis; document choice).
  - [ ] Persist **pairwise distances**, selected donor task, and **observed speed‑up** vs. from‑scratch.
  - [ ] Plot **similarity vs. transfer gain** → `experiments/figures/transfer.png`.
- [ ] **Curriculum — `experiments/run_curriculum.py`**
  - [ ] Cluster tasks by invariants (`β•`, spectral proxies, branching factor) into 3–4 **levels**.
  - [ ] Measure sample efficiency across levels; output `experiments/results/curriculum.csv`.
  - [ ] Plot curriculum learning curve → `experiments/figures/curriculum.png`.

---

## P2 — Correctness, Perf, and DevEx
- [ ] **Unit tests (pytest) — `tests/`**
  - [ ] `test_complex.py`: Laplacian shape, Fiedler vector exists; variance > 0 on asymmetric toy.
  - [ ] `test_mcts.py`: selection UCB monotonicity, rollout cap obeyed, run doesn’t crash on 3×3.
  - [ ] `test_metrics.py`: success/quality return expected values on trivial cases.
- [ ] **Performance**
  - [ ] Switch to `scipy.sparse.linalg.eigsh(k=2)` for Laplacian when grid ≥ 5×5; fallback to dense for tiny grids.
  - [ ] Memoize legal moves; early‑exit expansions when reward = 1.
- [ ] **Figures (matplotlib only)**
  - [ ] Small helper to plot bars/lines from CSVs; save under `experiments/figures/`.
- [ ] **Docs**
  - [ ] README: “Reproduce the paper” section with 2–3 commands (baselines, ablations, transfer).

---

## CLI & Config (what to expose)
- [ ] Global: `--seed`, `--trials`, `--iterations`, `--dmax_policy`, `--weights w_c w_d w_s`, `--grid_mix`, `--alphabet`, `--split {train,val,test}`.
- [ ] Output dirs: `--out_csv`, `--out_json`, `--figdir`.
- [ ] Method toggles: `--methods mcts_std,tmcts,…`

**Example:**
```bash
python experiments/run_baselines.py   --trials 5 --iterations 200   --methods mcts_std,tmcts   --weights 0.4 0.3 0.3   --out experiments/results/baselines.csv
```

---

## Migration notes
- [ ] Use `src/topomcts/legacy_port.py` and `notebooks/legacy.ipynb` only for **reference**; port stable logic into modules with docstrings.
- [ ] Keep public API minimal: `ARCGame`, `SimplicialComplex`, `TopologicalMCTSEngine`, `TopologicalMetaAgent`, `generate_dataset`, `metrics`.

---

## Definition of Done (sign off checklist)
- [ ] Fresh venv run produces **all** tables/figures in `experiments/results/` and `experiments/figures/`.
- [ ] Seeds + manifest guarantee identical numbers across machines.
- [ ] README updated with exact commands used for paper results.
- [ ] Tests pass locally; smoke run completes < 2 min on laptop.
