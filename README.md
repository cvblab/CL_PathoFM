# When Does Curriculum Learning Help Downstream Classification with Pathology Foundation Models? Evidence from Three Multi-Annotated Cohorts


This repository contains the code required to replicate the experiments described in the article ‘When Does Curriculum Learning Help Downstream Classification with Pathology Foundation Models? Evidence from Three Multi-Annotated Cohorts’


## 1. Expected directory layout

To regenerate Table I, Table II, Table III, Table IV, and Figure 2 from the manuscript, starting from
precomputed foundation-model embeddings, set up your project like this:
```
your_project/
├── github_code/              <- this folder
├── processed/                 <- AI4SKIN patch embeddings (CONCHv1.5/UNI2/VIRCHOW2 subfolders)
├── data_mhist/                <- MHIST patch embeddings
├── data_crowdgleason/         <- CrowdGleason patch embeddings
```
The scripts below will create folders under `results_*` in `your_project`.


## 2. Prerequisites

**Python**: `torch`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `seaborn`, `matplotlib`. A CUDA GPU is strongly recommended — a single non-MIL run
(MHIST/CrowdGleason) takes ~2–8 minutes at 50 epochs; a MIL run (AI4SKIN) somewhat longer; the full grids (hundreds of runs each) are impractical on
CPU.

**Embeddings**: this package does *not* include or generate FM embeddings. You need, per cohort, one `.pt`/`.npy` file per sample under
`processed/<ENCODER>/` (AI4SKIN, patch-bags) or `data_mhist/<ENCODER>/` / `data_crowdgleason/<ENCODER>/` (single embedding per patch), for each of
`CONCHv1.5`, `UNI2`, `VIRCHOW2`. Extract these with the paper's own foundation-model encoders before running anything here.

**CSV manifests**: included, at `src/csv/` (AI4SKIN), `src/csv_mhist/`, `src/csv_crowdgleason/` — each has `train.csv`/`val.csv`/`test.csv` with the
exact splits used in the paper.

## 3. Table I — main grid (AI4SKIN, MHIST, CrowdGleason)

Run once per cohort, and per schedule (`reorder`, `subsets`, `weights`) — each call trains all five variants (non-CL + 4 criteria) × 3 encoders × 5 seeds for that schedule.

```bash
cd github_code/src/external

# AI4SKIN — 
for STRATEGY in reorder subsets weights; do
  python run_external.py --dataset ai4skin --num-classes 6 \
    --csv-dir ../csv --data-root ../../../processed \
    --results-dir ../../../results_ai4skin_full \
    --strategy $STRATEGY
done

# MHIST — 
for STRATEGY in reorder subsets weights; do
  python run_external.py --dataset mhist --num-classes 2 --strategy $STRATEGY \
    --data-root ../../../data_mhist --results-dir ../../../results_mhist
done

# CrowdGleason — 
for STRATEGY in reorder subsets weights; do
  python run_external.py --dataset crowdgleason --num-classes 4 --strategy $STRATEGY \
    --data-root ../../../data_crowdgleason --results-dir ../../../results_crowdgleason
done
```

## 4. Table II — CrowdGleason baselines and controls

Runs every method in `METHODS` (non-CL, the 4 controls, 4 generic robust losses, soft labels, and Prior CL in both schedules) × 3 encoders × 5 seeds
= 195 runs, into `results_baselines_crowdgleason/logs/`. Use `--list` to see the method names, `--methods <name> [<name> ...]` to run a subset, or
`--dry-run` to preview without training. CrowdGleason data path is hardcoded (`DATASETS` dict, line ~84) to `/root/data_crowdgleason`.

```bash
cd github_code/src/external
python run_baselines.py --datasets crowdgleason \
  --results-root ../../../results_baselines_crowdgleason
```

## 5. Table III — Agreement-tier curriculum

Run both budget readings:

```bash
cd github_code/src/external
python run_wei_curriculum.py --budget total \
  --data-root ../../../data_mhist --results-root ../../../results_wei_mhist_total
python run_wei_curriculum.py --budget perstage \
  --data-root ../../../data_mhist --results-root ../../../results_wei_mhist_perstage
```

## 6. Table IV and Figure 2 — CrowdGleason Ablation

Two scripts, `src/experiment3.py` (label noise) and `src/experiment4.py` (data scarcity). Both need `--csv-dir`/`--data-root` pointed at CrowdGleason
explicitly and `--num-classes 4`.

```bash
cd github_code/src

# Noise (Table IV, both rows): 3 encoders x 2 levels x 5 criteria x 5 seeds = 150 runs
for DATASET in CONCHv1.5 UNI2 VIRCHOW2; do
  python experiment3.py --dataset $DATASET --strategy reorder --ordered \
    --noise-levels 0.05 0.30 --seeds 42 53 78 102 294 --epochs 50 \
    --results-root ../../results_rerun_noise_crowdgleason \
    --data-root ../../data_crowdgleason --csv-dir ./csv_crowdgleason --num-classes 4
done

# Scarcity (Table IV + Figure 2): 3 encoders x 4 fractions x 5 criteria x 5 seeds = 300 runs
for DATASET in CONCHv1.5 UNI2 VIRCHOW2; do
  python experiment4.py --dataset $DATASET --strategy subsets --ordered \
    --fractions 0.10 0.25 0.50 1.00 --seeds 42 53 78 102 294 --epochs 50 \
    --results-root ../../results_rerun_scarcity_crowdgleason \
    --data-root ../../data_crowdgleason --csv-dir ./csv_crowdgleason --num-classes 4
done
```

Then generate Figure 2:

```bash
cd github_code/src
python plot_scarcity_heatmap_crowdgleason.py redgreen    # red/yellow/green
```


## 7. Regenerating the tables themselves

`analysis/make_results.py` reads every `results_*` directory above and recomputes every number in Tables I, II, III, and IV (including the paired
significance tests and Bonferroni corrections) directly from the raw per-run JSON files, writing a Markdown report.

```bash
cd github_code
python analysis/make_results.py
```


## 8. Reproducibility notes

- All main-grid and stress-test runs use five independent runs `{42, 53, 78, 102, 294}` and 50 epochs.
- Every script skips (encoder, schedule/level, seed) combinations already completed on disk, so an interrupted run can simply be restarted.
- Optimizer settings differ by architecture, not by curriculum variant: AdamW (lr=1e-4, weight decay=1e-4, cosine-annealed) for the non-MIL
  MHIST/CrowdGleason classifier head; Adam (lr=1e-3, weight decay=1e-4, constant LR) for the AI4SKIN MIL pipeline (TransMIL aggregator + head,
  trained jointly from scratch each run). This is handled automatically by `train.py`/`experiment3.py`/`experiment4.py` — nothing to configure.



