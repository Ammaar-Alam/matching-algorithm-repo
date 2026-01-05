# PAM (Probabilistic Acceptability Model)

This folder adds a probabilistic, triad‑aware scoring pipeline consistent with the ROADMAP: soft acceptability per item, importance calibration, uncertainty‑aware LCB, and pairwise scoring.

## What’s included
- `models/accept_kernels.py` — Likert and MC acceptability kernels with monotone defaults.
- `models/importance_map.py` — simple monotone importance calibration map λ.
- `models/domain_weights.py` — domain multipliers and Top‑3 boost (ψ).
- `models/mc_similarity.py` — builds MC option similarity matrices from acceptable sets.
- `scoring/directional.py` — directional satisfaction ŝ, variance, n_eff; LCB helper.
- `scoring/pair_score.py` — combines directional LCBs via additive logits.
- `scoring/score_pairs_cli.py` — emits scores for all ordered pairs (A,B).
- `eval/evaluate_pam.py` — couples vs random evaluation (AUC, lift).
- `matching/rankings.py` — builds incomplete ranked lists from scores.
- `matching/stable.py` — simple Gale–Shapley matching on lists.

## Quick start
Assuming you already produced `export.csv`, `couples.csv`, and `tiger-alg/meta.json` as in the repo README:

```bash
cd tiger-alg
python scoring/score_pairs_cli.py \
  --export_csv ../export.csv \
  --meta_json meta.json \
  --questions_json ../data/questions.json \
  --delta 0.10 \
  --out scores.csv

python eval/evaluate_pam.py \
  --export_csv ../export.csv \
  --couples_csv ../couples.csv \
  --meta_json meta.json \
  --questions_json ../data/questions.json \
  --delta 0.10 \
  --out out/summary_pam.csv
```

To derive ranked lists and a toy bipartite matching (for demos):
```bash
python matching/rankings.py --scores_csv scores.csv --tau 0.20 --out lists.json
python matching/stable.py --lists lists.json --mode bipartite --out matching.json
```

## Notes
- Kernels come with sensible monotone defaults; training can later refine parameters.
- MC similarities are built from the survey’s acceptable sets using Jaccard‑like co‑acceptance.
- LCB uses a closed‑form normal approximation with common quantiles for δ∈{0.05,0.10,0.20}.

## Training (domain weights via pairwise logistic)

```bash
python train/trainer.py \
  --export_csv ../export.csv \
  --couples_csv ../couples.csv \
  --meta_json meta.json \
  --questions_json ../data/questions.json \
  --outdir out_pam \
  --negatives 20 --delta 0.10 --seed 1

# Use learned weights for scoring/eval
python scoring/score_pairs_cli.py \
  --export_csv ../export.csv \
  --meta_json meta.json \
  --questions_json ../data/questions.json \
  --weights_json out_pam/weights.json \
  --delta 0.10 \
  --out scores.csv

python eval/evaluate_pam.py \
  --export_csv ../export.csv \
  --couples_csv ../couples.csv \
  --meta_json meta.json \
  --questions_json ../data/questions.json \
  --weights_json out_pam/weights.json \
  --delta 0.10 \
  --out out/summary_pam.csv
```

Optional evaluation extras:

```bash
python eval/evaluate_pam.py \
  --export_csv ../export.csv \
  --couples_csv ../couples.csv \
  --meta_json meta.json \
  --questions_json ../data/questions.json \
  --delta 0.10 \
  --bootstrap 500 --permute 200 \
  --out out/summary_pam.csv
```

## Merged IDF+LCB pipeline (ROADMAP)

For a small‑n friendly, IDF‑weighted, LCB‑penalized scoring and matching pipeline described in `ROADMAP.md`:

```bash
cd tiger-alg
python matching/merged_pipeline.py \
  --export_csv ../export.csv \
  --couples_csv ../couples.csv \
  --meta_json meta.json \
  --outdir out_merged \
  --rho 0.35 --delta 0.10 --tau 0.05 --mu 0.03 --topk 2
```

This writes:
- `out_merged/pair_scores_merged.csv` — harmonic‑mean, reciprocity‑aware pair scores with LCBs.
- `out_merged/lists_tied.json` — per‑person preference lists with ties (SMTI‑style).
- `out_merged/matching_da.json` — bipartite DA outcome (stable under a simple split).
- `out_merged/matching_best_stable_lp.json` — maximum‑weight stable matching via MILP (falls back to DA if MILP is unavailable).
- `out_merged/matching_max_weight.json` — exact max‑weight matching (blossom, via NetworkX, with greedy fallback).
- `out_merged/summary_ranks.json` — per‑person ranks, top‑gaps, robustness tags, and Hit@k / median rank summary.

*(AI-Disclaimer: This documentation was written and/or generated with the help of AI; please verify all claims and ensure accuracy before use)*
