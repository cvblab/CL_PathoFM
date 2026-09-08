"""Experiment 4 -- performance under data scarcity (AI4SKIN).

    python experiment4.py --dataset VIRCHOW2 --strategy subsets --ordered \
        --fractions 0.10 0.25 0.50 --seeds 42 53 78 102 294 --epochs 50 \
        --results-root ../results_rerun_scarcity --data-root ./processed
"""

import argparse, json, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

from models import build_model
from train import run_training_experiment
from utils import detect_input_dim_and_mil, set_seed

EXPERIMENTS = ["non_cl", "cl_prior", "cl_adaptive_raw",
               "cl_combined_staged", "cl_combined_optimized"]


def subsample(csv_path, frac, out_dir, seed):
    """Stratified subsample of the training split, preserving class balance."""
    df = pd.read_csv(csv_path)
    if frac >= 1.0:
        return csv_path
    out = (df.groupby("GT", group_keys=False)
             .apply(lambda g: g.sample(n=max(1, int(round(len(g) * frac))),
                                       random_state=seed)))
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"train_frac{int(frac*100)}_seed{seed}.csv")
    out.to_csv(p, index=False)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="VIRCHOW2",
                    choices=["CONCHv1.5", "UNI2", "VIRCHOW2"])
    ap.add_argument("--strategy", default="subsets",
                    choices=["baseline", "reorder", "subsets", "weights"])
    ap.add_argument("--ordered", action="store_true")
    ap.add_argument("--fractions", nargs="+", type=float,
                    default=[0.10, 0.25, 0.50, 1.00])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 53, 78, 102, 294])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--results-root", default="./results")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--csv-dir", default=None)
    ap.add_argument("--num-classes", type=int, default=6,
                    help="Number of GT classes (AI4SKIN=6, CrowdGleason=4)")
    a = ap.parse_args()

    tag = a.strategy + ("_ordered" if a.ordered else "")
    results_path = os.path.join(a.results_root, f"experiment4_{a.dataset}_{tag}")
    os.makedirs(results_path, exist_ok=True)

    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(a.data_root) if a.data_root else os.path.abspath(os.path.join(here, "../processed"))
    csv_dir = os.path.abspath(a.csv_dir) if a.csv_dir else os.path.join(here, "csv")
    base_train = os.path.join(csv_dir, "train.csv")
    csv_val, csv_test = (os.path.join(csv_dir, f"{s}.csv") for s in ("val", "test"))
    folder = os.path.join(data_dir, a.dataset)

    dim, is_mil = detect_input_dim_and_mil(base_train, folder)
    if dim == 0:
        raise SystemExit(f"[SKIP] no readable embeddings in {folder}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    for frac in a.fractions:
        for exp in EXPERIMENTS:
            f1s = []
            for r, seed in enumerate(a.seeds):
                set_seed(seed)
                train_csv = subsample(base_train, frac,
                                      os.path.join(results_path, "frac_csvs"), seed)
                stem = os.path.join(results_path, f"tmp_{exp}_frac{int(frac*100)}_run{r}")
                lj, tj = stem + ".json", stem + "_test.json"
                if os.path.exists(tj):
                    f1s.append(json.load(open(tj))["f1"]); continue

                print(f"--- {a.dataset} | frac {frac:.2f} | {exp} | seed {seed}", flush=True)
                model = build_model(dim, a.num_classes, is_mil).to(device)
                if not is_mil:
                    opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
                    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
                else:
                    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
                    sch = None
                try:
                    _, f1 = run_training_experiment(
                        csv_train=train_csv, folder=folder, model=model,
                        optimizer=opt, criterion=nn.CrossEntropyLoss(), device=device,
                        is_mil=is_mil, experiment_name=exp, pacing="none", seed=seed,
                        val_csv=csv_val, test_csv=csv_test, epochs=a.epochs, n_bins=5,
                        metrics_log_path=lj, test_metrics_path=tj,
                        cl_mode="standard", hybrid_prob=0.0,
                        strategy=("baseline" if exp == "non_cl" else a.strategy),
                        ordered_cl=a.ordered, scheduler=sch)
                    f1s.append(f1)
                except Exception as e:
                    import traceback; print(f"[ERROR] {exp}/{frac}/{seed}: {e}"); traceback.print_exc()
            if f1s:
                rows.append({"Data Fraction": frac * 100, "Strategy": exp,
                             "Clean Test F1": float(np.mean(f1s)),
                             "Std_F1": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
                             "N_runs": len(f1s)})
                pd.DataFrame(rows).to_csv(
                    os.path.join(results_path, "data_scarcity_results.csv"), index=False)

    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nSaved {results_path}/data_scarcity_results.csv")


if __name__ == "__main__":
    main()
