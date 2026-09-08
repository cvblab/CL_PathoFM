"""
Run the existing curriculum experiments against an external dataset.

This is `main.py` with the three things it hardcodes made into flags:
`NUM_CLASSES`, the data root, and the CSV directory.

-----
    python run_external.py --dataset mhist --num-classes 2
    python run_external.py --dataset crowdgleason --num-classes 4
"""

import argparse
import json
import os
import sys

import pandas as pd
import torch
from torch import nn, optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model                     # noqa: E402
from train import run_training_experiment          # noqa: E402
from utils import detect_input_dim_and_mil, get_available_datasets, set_seed  # noqa: E402

SEEDS = [42, 53, 78, 102, 294]
PACINGS = ["linear", "exponential", "inverse", "step"]
ALL_EXPERIMENTS = ["non_cl", "cl_prior", "cl_adaptive_raw",
                   "cl_combined_staged", "cl_combined_optimized"]
# Everything except non_cl and cl_adaptive_raw needs annotator disagreement.
ADAPTIVE_ONLY = ["non_cl", "cl_adaptive_raw"]


def has_marker_columns(csv_path):
    cols = pd.read_csv(csv_path, nrows=1).columns
    return any(c.startswith("Marker_") for c in cols)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True,
                   help="Tag used to locate ../csv_<tag> and ../../data_<tag>, and to name results")
    p.add_argument("--num-classes", type=int, required=True)
    p.add_argument("--csv-dir", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--results-dir", default=None)
    p.add_argument("--encoders", nargs="*", default=None,
                   help="Subset of encoder folders under --data-root (default: all present)")
    p.add_argument("--experiments", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--strategy", default="reorder",
                   choices=["baseline", "reorder", "subsets", "weights"])
    p.add_argument("--anti-curriculum", action="store_true")
    p.add_argument("--hybrid-prob", type=float, default=0.0)
    args = p.parse_args()

    csv_dir = args.csv_dir or os.path.join(here, "..", f"csv_{args.dataset}")
    data_root = args.data_root or os.path.join(here, "..", "..", f"data_{args.dataset}")
    results_dir = args.results_dir or os.path.join(here, "..", "..", f"results_{args.dataset}")
    logs_dir = os.path.join(results_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    csv_train = os.path.join(csv_dir, "train.csv")
    csv_val = os.path.join(csv_dir, "val.csv")
    csv_test = os.path.join(csv_dir, "test.csv")

    experiments = args.experiments
    if experiments is None:
        experiments = ALL_EXPERIMENTS if has_marker_columns(csv_train) else ADAPTIVE_ONLY
        if experiments is ADAPTIVE_ONLY:
            print("[INFO] No Marker_ columns found -> restricting to "
                  f"{ADAPTIVE_ONLY} (prior difficulty is undefined here).")

    encoders = args.encoders or get_available_datasets(data_root)
    if not encoders:
        raise SystemExit(f"No encoder folders under {data_root}. Run the prep script first.")
    print(f"[INFO] dataset={args.dataset} encoders={encoders} experiments={experiments}")

    cl_mode = "anti" if args.anti_curriculum else "standard"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for encoder_name in encoders:
        folder = os.path.join(data_root, encoder_name)
        input_dim, is_mil = detect_input_dim_and_mil(csv_train, folder)
        if input_dim == 0:
            print(f"[SKIP] no readable embeddings in {folder}")
            continue
        print(f"\n[ENCODER] {encoder_name} | dim {input_dim} | MIL {is_mil}")

        for seed in args.seeds:
            set_seed(seed)
            for experiment_name in experiments:
                if experiment_name == "non_cl":
                    pacings = ["none"]
                elif args.strategy != "baseline":
                    pacings = ["none"]
                else:
                    pacings = PACINGS

                for pacing in pacings:
                    tag = (f"{experiment_name}_pacing-{pacing}_dataset-{encoder_name}"
                           f"_seed-{seed}_{args.strategy}")
                    metrics_path = os.path.join(logs_dir, f"{tag}.json")
                    test_path = metrics_path.replace(".json", "_test.json")
                    if os.path.exists(test_path):
                        print(f"[SKIP] done: {tag}")
                        continue

                    print(f"--> {tag}")
                    model = build_model(input_dim, args.num_classes, is_mil).to(device)
                    if not is_mil:
                        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
                        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
                    else:
                        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
                        scheduler = None

                    try:
                        run_training_experiment(
                            csv_train=csv_train, folder=folder, model=model,
                            optimizer=optimizer, criterion=nn.CrossEntropyLoss(),
                            device=device, is_mil=is_mil,
                            experiment_name=experiment_name, pacing=pacing, seed=seed,
                            val_csv=csv_val, test_csv=csv_test, epochs=args.epochs,
                            metrics_log_path=metrics_path, test_metrics_path=test_path,
                            cl_mode=cl_mode, hybrid_prob=args.hybrid_prob,
                            strategy=args.strategy, scheduler=scheduler,
                        )
                    except Exception as exc:
                        print(f"[ERROR] {tag}: {exc}")
                        import traceback
                        traceback.print_exc()

    # Rebuild the aggregate table from whatever is on disk, same shape as main.py's.
    rows = []
    for fn in os.listdir(logs_dir):
        if not fn.endswith("_test.json"):
            continue
        stem = fn[: -len("_test.json")]
        parts = stem.split("_dataset-")
        exp_pacing, rest = parts[0], parts[1]
        experiment, pacing = exp_pacing.split("_pacing-")
        encoder, _, tail = rest.partition("_seed-")
        seed, _, strategy = tail.partition("_")
        with open(os.path.join(logs_dir, fn)) as f:
            data = json.load(f)
        rows.append({"encoder": encoder, "experiment": experiment, "pacing": pacing,
                     "strategy": strategy, "seed": int(seed),
                     "acc": data.get("accuracy", 0.0), "f1": data.get("f1", 0.0)})

    if rows:
        df = pd.DataFrame(rows).sort_values(["encoder", "experiment", "pacing", "seed"])
        df.to_csv(os.path.join(results_dir, "final_results_raw.csv"), index=False)
        agg = (df.groupby(["encoder", "experiment", "pacing", "strategy"])
                 .agg(acc_mean=("acc", "mean"), acc_std=("acc", "std"),
                      f1_mean=("f1", "mean"), f1_std=("f1", "std")).reset_index())
        agg["acc"] = agg.apply(lambda r: f"{r.acc_mean:.3f} ± {r.acc_std:.3f}", axis=1)
        agg["f1"] = agg.apply(lambda r: f"{r.f1_mean:.3f} ± {r.f1_std:.3f}", axis=1)
        out = agg[["encoder", "experiment", "pacing", "strategy", "acc", "f1"]]
        out.to_csv(os.path.join(results_dir, "final_results_mean.csv"), index=False)
        with open(os.path.join(results_dir, "final_results_mean.tex"), "w") as f:
            f.write(out.to_latex(index=False))
        print(f"\n[INFO] aggregated {len(df)} runs into {results_dir}")


if __name__ == "__main__":
    main()
