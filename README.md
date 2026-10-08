# Are Attention Weights Explanations?

Reference implementation for the paper

> **Are Attention Weights Explanations? A Ground-Truth-Anchored Faithfulness and
> Stability Audit of Tabular Attention Models for Employee Burnout and Turnover Risk**
> H. Singh and B. Singh, *International Journal of Intelligent Engineering and Systems* (IJIES).

This repository implements every algorithm in the paper and regenerates **Tables 3–5**
and **Figures 2–3** of the results section:

* **ARCA** — Attention-Routed Contrastive Attribution with the exact value/routing
  decomposition (Algorithm 1, Eqs 9–12), for FT-Transformer and TabNet.
* The two **attention models** (FT-Transformer, TabNet) with their native readouts
  ([CLS] attention, rollout, gradient-weighted attention, TabNet masks).
* The five **reference models** (LR, MLP, EBM, XGBoost, LightGBM) and their SHAP.
* **Oracle attributions** recovered from each generator (linear Shapley for R1/R4,
  exact interventional Shapley for R2, TreeSHAP surrogate for R3).
* The **E1–E5 evaluation protocol**: oracle agreement (Kendall τ, precision@|P|),
  deletion/insertion faithfulness, counterfactual faithfulness, seed stability,
  local Lipschitz, the Redundancy-Split Index, and Kendall’s W replication.

```
src/arca/
  data/         synthetic regime generators (R1–R4) + Kaggle loaders + manifests
  models/       ft_transformer.py, tabnet.py, references.py (+ trainer)
  attributions/ arca.py, oracles.py, shap_est.py, attention_readouts.py
  evaluation/   metrics.py (E1–E5), protocol helpers
  figures/      make_tables.py (Tables 3–5), make_figures.py (Figs 2–3)
  experiments/  run_task.py, run_all.py
results/
  reported/     published values from the paper (used to regenerate tables/figures)
  tables/       generated table3/4/5 .csv .tex .png
  figures/      generated figure2/figure3 .png
tests/          test_arca_completeness.py, test_metrics.py
configs/        tasks.yaml (splits, tuning budgets, search spaces, seeds)
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # full stack (torch, xgboost, …)
# or, to only regenerate tables/figures and run the NumPy tests:
pip install numpy pandas scipy scikit-learn matplotlib
```

## Quick start — regenerate the result tables and figures

No data download needed; this reads the published values in `results/reported/`:

```bash
export PYTHONPATH=src
python -m arca.figures.make_tables     # -> results/tables/table3|4|5 .csv .tex .png
python -m arca.figures.make_figures    # -> results/figures/figure2|figure3 .png
```

## Run the full pipeline (synthetic regimes, no download)

The synthetic generators reproduce the four regimes so the whole pipeline runs
end to end:

```bash
export PYTHONPATH=src
python -m arca.experiments.run_task --task T1            # one task
python -m arca.experiments.run_all  --out results/runs   # all tasks + tables/figures
# add --no-torch to run references + oracle + SHAP only (skips the nets/ARCA)
```

## Reproduce the reported numbers on the real data

1. Download the three public Kaggle datasets into `data/raw/`:
   * D1 — *Mental Health & Burnout in Tech Workers 2026* (CC0)
   * D2 — *Employee Burnout & Turnover Prediction* (CC BY-NC-ND 4.0; not redistributed)
   * D3 — *Work From Home Employee Burnout* (CC0)
2. The loaders in `src/arca/data/loaders.py` apply the leakage-exclusion manifests
   (`manifests.py`, Table 2) and build the task labels.
3. Run `run_all.py --real` with the seeds and splits in `configs/tasks.yaml`.

> The generator-recovery R² values (T1 = 0.714, T4 = 0.410, T5 = 0.922; R2 = 0.998)
> make clear that the T1/T4/T5 oracles are **fitted surrogates**: agreement with them
> is agreement with a benchmark, not with the true data-generating process.

## Validate ARCA’s identities

`tests/test_arca_completeness.py` checks, in pure NumPy, that the ARCA
decomposition is exact:

```
[C1] completeness ........ sum(total)  = f(x) - f(x_bar)
[C2] exact split ......... total = value + routing       (machine precision)
[C3] value completeness .. sum(value)  = f(x) - f(x_bar; A = A(x))
     routing meaning ...... sum(routing)= f(x_bar; A(x)) - f(x_bar)
[C4] linear reduction .... routing = 0 and total = interventional Shapley
```

```bash
python tests/test_arca_completeness.py
python tests/test_metrics.py
# or:  pytest -q
```

## How ARCA works (Algorithm 1)

ARCA integrates the logit gradient along a straight path from the background-mean
reference `x_bar` to the instance `x` (an integrated-gradients estimate, Eq 9).
The **routing recorded at the instance** — the per-head attention probabilities and
LayerNorm statistics (FT-Transformer) or the step masks (TabNet) — is then *replayed*
as a constant to obtain the **value channel** (Eq 10); the **routing channel** is the
remainder (Eq 11). The per-instance **routing share** says whether attention can be
read through its value paths (low share) or itself takes the decision (share near one).

## Citation

See `CITATION.cff`. Code released under the MIT License (`LICENSE`).
