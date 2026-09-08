import os
import argparse
import time
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from torch import nn, optim
import torch

from train import run_training_experiment
from models import build_model
from utils import detect_input_dim_and_mil, set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Run Label Noise Robustness (Experiment 3)")
    parser.add_argument("--strategy", type=str, default="reorder", choices=["baseline", "reorder", "subsets", "weights"], help="Curriculum Strategy")
    parser.add_argument("--dataset", type=str, default="UNI2", choices=["CONCHv1.5", "UNI2", "VIRCHOW2"], help="Dataset / Encoder")
    parser.add_argument("--ordered", action="store_true", help="Use Ordered Curriculum (Strict Sort)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of max epochs")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per experiment (mean calculation)")
    parser.add_argument("--noise-levels", nargs="+", type=float, default=None,
                        help="Asymmetric noise fractions, e.g. 0.05 0.30")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Explicit seed list; default is 42..42+runs-1")
    parser.add_argument("--results-root", type=str, default="./results",
                        help="Where to write experiment3_<enc>_<strategy> output")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Embedding root; default ../processed relative to this script")
    parser.add_argument("--csv-dir", type=str, default=None,
                        help="CSV dir; default ./csv relative to this script")
    parser.add_argument("--num-classes", type=int, default=6,
                        help="Number of GT classes (AI4SKIN=6, CrowdGleason=4)")
    return parser.parse_args()

def inject_asymmetric_noise(csv_path, noise_level=0.2, out_dir=".", seed=42, num_classes=6):
    """
    Inyecta ruido simulando errores clínicos reales (confusión entre clases adyacentes).
    Asume clases 0 a num_classes-1.
    """
    np.random.seed(seed)
    df = pd.read_csv(csv_path)
    noisy_df = df.copy()

    n_samples = len(df)
    n_noisy = int(n_samples * noise_level)
    noisy_indices = np.random.choice(df.index, n_noisy, replace=False)
    max_class = num_classes - 1

    for idx in noisy_indices:
        gt = df.at[idx, 'GT']
        possible_flips = [c for c in [gt-1, gt+1] if 0 <= c <= max_class]
        if possible_flips:
            noisy_df.at[idx, 'GT'] = np.random.choice(possible_flips)
            
    os.makedirs(out_dir, exist_ok=True)
    noisy_path = os.path.join(out_dir, f"train_asym_noise_{int(noise_level*100)}_seed{seed}.csv")
    noisy_df.to_csv(noisy_path, index=False)
    return noisy_path

def main():
    args = parse_args()
    
    strategy_tag = f"{args.strategy}"
    if args.ordered:
        strategy_tag += "_ordered"
        
    dataset = args.dataset
    results_path = os.path.join(args.results_root, f"experiment3_{dataset}_{strategy_tag}")
    os.makedirs(results_path, exist_ok=True)
    
    noise_levels = args.noise_levels if args.noise_levels else [0.10]
    experiments = ["non_cl", "cl_prior", "cl_adaptive_raw", "cl_combined_staged", "cl_combined_optimized"]
    
    runs = args.runs
    epochs = args.epochs
    seed_list = args.seeds if args.seeds else [42 + r for r in range(runs)]
    runs = len(seed_list)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(args.data_root) if args.data_root else os.path.abspath(os.path.join(script_dir, "../processed"))
    csv_dir = os.path.abspath(args.csv_dir) if args.csv_dir else os.path.abspath(os.path.join(script_dir, "csv"))
    
    base_train_csv = os.path.join(csv_dir, "train.csv")
    csv_val = os.path.join(csv_dir, "val.csv")
    csv_test = os.path.join(csv_dir, "test.csv")
    folder = os.path.join(data_dir, dataset)

    input_dim, is_mil = detect_input_dim_and_mil(base_train_csv, folder)
    if input_dim == 0:
        print(f"[SKIP] Could not detect valid data for {dataset}")
        return

    all_results = []

    for level in noise_levels:
        for exp_name in experiments:
            print(f"--- Evaluando {exp_name} | Noise Level {level*100}% ---")
            
            run_f1_scores = []
            
            for r in range(runs):
                set_seed(seed_list[r])
                
                noise_dir = os.path.join(results_path, "noisy_csvs")
                train_csv = inject_asymmetric_noise(base_train_csv, noise_level=level, out_dir=noise_dir, seed=seed_list[r], num_classes=args.num_classes) if level > 0 else base_train_csv

                metrics_log_path = os.path.join(results_path, f"tmp_{exp_name}_noise{int(level*100)}_run{r}.json")
                test_metrics_path = os.path.join(results_path, f"tmp_{exp_name}_noise{int(level*100)}_run{r}_test.json")
                model_best_path = metrics_log_path.replace(".json", "_best.pt")

                skip_training = False
                if os.path.exists(metrics_log_path) and os.path.exists(test_metrics_path):
                    try:
                        with open(metrics_log_path, "r") as f:
                            metrics_log = json.load(f)
                        if len(metrics_log) >= epochs:
                            print(f"      -> Skipped: Log found for run {r} on noise {level*100}%")
                            skip_training = True
                            with open(test_metrics_path, "r") as f:
                                test_metrics = json.load(f)
                                run_f1_scores.append(test_metrics["f1"])
                    except Exception:
                        pass

                if skip_training:
                    continue

                model = build_model(input_dim, args.num_classes, is_mil)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device)

                if not is_mil:
                    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                else:
                    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
                    scheduler = None
                criterion = nn.CrossEntropyLoss()

                current_strategy = "baseline" if exp_name == "non_cl" else args.strategy

                start_train = time.time()
                test_acc, test_f1 = run_training_experiment(
                    csv_train=train_csv,
                    folder=folder,
                    model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    is_mil=is_mil,
                    experiment_name=exp_name,
                    pacing="none",
                    seed=seed_list[r],
                    val_csv=csv_val,
                    test_csv=csv_test,
                    epochs=epochs,
                    n_bins=5,
                    metrics_log_path=metrics_log_path,
                    test_metrics_path=test_metrics_path,
                    resume_checkpoint=None,
                    cl_mode="standard",
                    hybrid_prob=0.0,
                    strategy=current_strategy,
                    ordered_cl=args.ordered,
                    scheduler=scheduler
                )
                
                run_f1_scores.append(test_f1)

            # Record mean statistics across runs
            all_results.append({
                "Noise Level": f"{int(level*100)}%",
                "Noise_Raw": level,
                "Strategy": exp_name,
                "Clean Test F1": np.mean(run_f1_scores),
                "Std_F1": np.std(run_f1_scores)
            })

    df_res = pd.DataFrame(all_results)
    noise_csv = os.path.join(results_path, "noise_robustness.csv")
    df_res.to_csv(noise_csv, index=False)
    print(f"\nSaved noise robustness results to {noise_csv}")
    print("\nSummary:")
    print(df_res)

    # Plot robustez
    plt.figure(figsize=(9, 6))
    
    sns.barplot(data=df_res, x="Noise Level", y="Clean Test F1", hue="Strategy", edgecolor='black')
    
    plt.title("Robustness to Clinical (Asymmetric) Label Noise", fontsize=14)
    plt.ylabel("F1 Score on Clean Test Set", fontsize=12)
    plt.xlabel("Label Noise Level (%)", fontsize=12)
    plt.ylim(0.0, 1.0) # Depending on F1 scores, could be higher
    plt.legend(title="Training Strategy")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    pdf_path = os.path.join(results_path, "clinical_noise_robustness.pdf")
    plt.savefig(pdf_path)
    print(f"Saved plot to {pdf_path}")

if __name__ == "__main__":
    main()
