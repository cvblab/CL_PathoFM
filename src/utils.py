
import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from dataloader import AI4SKIN

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_available_datasets(data_dir="../data"):
    """
    Returns a sorted list of all subdirectories in data_dir.
    """
    if not os.path.exists(data_dir):
        return []
    return sorted([
        d for d in os.listdir(data_dir) 
        if os.path.isdir(os.path.join(data_dir, d))
    ])

def detect_input_dim_and_mil(csv_file, folder):
    """
    Detects input dimension and whether the dataset is MIL (2D) or not (1D).
    """
    # Create a temporary dataset to load the first item
    try:
        tmp_dataset = AI4SKIN(csv_file, folder)
        if len(tmp_dataset) == 0:
            raise ValueError(f"Dataset at {folder} is empty or CSV {csv_file} has no valid entries.")
        
        # Try loading with and without prefix
        try:
             x0, _ = tmp_dataset[0]
        except FileNotFoundError:
             pass
        

        x0, _ = tmp_dataset[0]

        if x0.dim() == 2:
            is_mil = True
            input_dim = x0.shape[1]
        else:
            is_mil = False
            input_dim = x0.shape[-1]
            
        return input_dim, is_mil
    except Exception as e:
        print(f"[WARNING] Could not detect dim/MIL for {folder}: {e}")
        return 0, False

def save_metrics(metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

def plot_dif(difficulty, title, base_dir="./src"):
    """
    Plots a histogram of difficulty scores.
    """
    plt.figure(figsize=(6, 4))
    sns.histplot(difficulty, bins=20, kde=True)
    plt.title(title)
    plt.xlabel("Difficulty")
    plt.ylabel("Count")
    plt.tight_layout()
    
    save_dir = os.path.join(base_dir, "plots")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/{title}.png")
    plt.close()

def plot_difficulty_histograms_per_class(difficulty, labels, title, save_path):
    """
    Plots difficulty histograms separated by class.
    """
    df_temp = {"difficulty": difficulty, "label": labels}
    import pandas as pd
    df = pd.DataFrame(df_temp)
    
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x="difficulty", hue="label", palette="tab10", kde=True, bins=20, multiple="stack")
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def save_per_class_histogram(difficulty, labels, mode, epoch, base_dir="./src"):
    """
    Wrapper to save per-class histogram with a standard naming convention.
    """
    save_dir = os.path.join(base_dir, "plots/per_class")
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{mode}_epoch{epoch+1}.png"
    plot_difficulty_histograms_per_class(
        difficulty,
        labels,
        title=f"Difficulty per Class — {mode} — Epoch {epoch+1}",
        save_path=save_path,
    )

def plot_performance_vs_ratio(ratios, performances, metric_name, save_path):
    """
    Plots performance vs data ratio/progress.
    """
    plt.figure(figsize=(6, 4))
    plt.plot(ratios, performances, marker='o')
    plt.xlabel("Data Ratio / Progress")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} vs Progress")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
