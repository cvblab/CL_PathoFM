# train.py

import torch
import torch.nn as nn
import numpy as np
import json
import os
import math

from torch.utils.data import DataLoader, Subset
from dataloader import AI4SKIN
from utils import save_metrics, plot_dif, plot_performance_vs_ratio, save_per_class_histogram
from test import evaluate_model
from curriculum import (
    get_curriculum_indices,
    get_strategy_batch,
    compute_prior_difficulty,
    compute_adaptive_difficulty,
    compute_adaptive_difficulty_laplace,
    apply_pacing
)

# ---------------------------------------------------------
# MIL collate → bags of variable length
# ---------------------------------------------------------
def mil_collate(batch):
    xs = [item[0] for item in batch]   # list of bags, each (N_i, dim)
    ys = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return xs, ys

def train_epoch(model, dataloader, optimizer, criterion, device, is_mil, sample_weights=None):
    model.train()
    running_loss = 0.0
    
    # Check if we need weighted loss
    use_weights = sample_weights is not None
    if use_weights:
        weighted_criterion = nn.CrossEntropyLoss(reduction='none')
    
    batch_idx = 0
    
    for x, y in dataloader:
        if is_mil:
            x = x[0].to(device) # bag -> tensor
        else:
            x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        
        if use_weights:

            batch_size = y.size(0)
            start = batch_idx
            end = batch_idx + batch_size
            
            # Safety check
            if end > len(sample_weights):
                print(f"[WARN] Batch indices {start}:{end} exceed weights length {len(sample_weights)}")
                w = torch.ones(batch_size).to(device)
            else:
                w = torch.tensor(sample_weights[start:end], dtype=torch.float32).to(device)
            
            raw_loss = weighted_criterion(logits, y)
            loss = (raw_loss * w).mean()
            
            batch_idx += batch_size
        else:
            loss = criterion(logits, y)
            
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    return running_loss / max(1, len(dataloader))

def run_training_experiment(
    csv_train, folder, model, optimizer, criterion, device, is_mil,
    experiment_name, pacing="none", seed=42,
    val_csv="./csv/val.csv", test_csv="./csv/test.csv",
    epochs=20, n_bins=5,

    metrics_log_path="./logs/tmp.json",
    test_metrics_path="./logs/tmp_test.json",
    resume_checkpoint=None,
    cl_mode="standard",
    hybrid_prob=0.0,
    strategy="baseline",
    ordered_cl=False,
    scheduler=None
):
    
    # -----------------------------------------------------
    # Dataset Load
    # -----------------------------------------------------
    dataset_full = AI4SKIN(csv_train, folder)

    # --- Resume Checkpoint ---
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"[INFO] Resuming training from checkpoint: {resume_checkpoint}")
        try:
            model.load_state_dict(torch.load(resume_checkpoint))
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint: {e}")


    # -----------------------------------------------------
    # Difficulty Initialization
    # -----------------------------------------------------
    difficulty_mode = experiment_name
    
    # For Non-CL (baseline), we just run standard training
    if experiment_name == "non_cl":
         # Standard Shuffle Training
         if is_mil:
            dataloader = DataLoader(dataset_full, batch_size=1, shuffle=True, collate_fn=mil_collate)
         else:
            dataloader = DataLoader(dataset_full, batch_size=32, shuffle=True)
         
         best_f1 = 0.0
         best_state = None
         metrics_log = []
         
         for epoch in range(epochs):
             loss = train_epoch(model, dataloader, optimizer, criterion, device, is_mil)
             val_acc, val_f1, _, _, _ = evaluate_model(model, val_csv, folder, device)
             
             metrics_log.append({
                 "epoch": epoch + 1,
                 "loss": float(loss),
                 "accuracy": float(val_acc),
                 "f1": float(val_f1)
             })

             if val_f1 > best_f1:
                 best_f1 = val_f1
                 best_state = model.state_dict()
                 
             if scheduler:
                 scheduler.step()
                 
             # No Patience Check
                 
         save_metrics(metrics_log, metrics_log_path)
         if best_state:
             model.load_state_dict(best_state)
             torch.save(best_state, metrics_log_path.replace(".json", "_best.pt"))

         test_acc, test_f1, y_true, y_pred, y_prob = evaluate_model(model, test_csv, folder, device)
         save_metrics({"accuracy": float(test_acc), "f1": float(test_f1)}, test_metrics_path)
         
         ratios = [m["epoch"] / epochs for m in metrics_log]
         f1s = [m["f1"] for m in metrics_log]
         plot_performance_vs_ratio(ratios, f1s, "Validation F1", metrics_log_path.replace(".json", "_f1_plot.png"))
         
         return test_acc, test_f1

    # -----------------------------------------------------
    # CL Setup (Or Strategy Setup)
    # -----------------------------------------------------
    print(f"[INFO] Computing initial difficulty for {experiment_name}...")
    
    prior = compute_prior_difficulty(csv_train)
    adaptive = np.zeros_like(prior)

    needs_adaptive_start = experiment_name in ["cl_adaptive_raw", "cl_adaptive_laplace", "cl_combined_staged", "cl_combined_optimized"]
    
    if needs_adaptive_start:
         if experiment_name == "cl_adaptive_laplace":
             adaptive = compute_adaptive_difficulty_laplace(model, csv_train, folder, device)
         else:
             adaptive = compute_adaptive_difficulty(model, csv_train, folder, device)

    # Plot initial difficulty
    if experiment_name == "cl_prior":
        base_diff = prior
    elif experiment_name in ["cl_adaptive_raw", "cl_adaptive_laplace"]:
        base_diff = adaptive
    else: # combined or default
        base_diff = 0.5 * prior + 0.5 * adaptive
    
    dataset_name = os.path.basename(folder)
    title = f"initial_{dataset_name}_{experiment_name}_{strategy}_{seed}"
    results_root = os.path.dirname(os.path.dirname(metrics_log_path))
    plot_dif(base_diff, title, base_dir=results_root)

    best_f1 = 0.0
    best_state = None
    metrics_log = []
    
    staged_warmup = 0.30
    staged_transition = 0.70

    for epoch in range(epochs):
        progress = (epoch + 1) / epochs
        
        # --- Update Adaptive Difficulty ---
        if experiment_name in ["cl_adaptive_raw", "cl_adaptive_laplace", "cl_combined_staged", "cl_combined_optimized"]:
            if experiment_name == "cl_adaptive_laplace":
                 adaptive = compute_adaptive_difficulty_laplace(model, csv_train, folder, device)
            else:
                 adaptive = compute_adaptive_difficulty(model, csv_train, folder, device)

        # --- Calculate Current Epoch Difficulty ---
        if experiment_name == "cl_prior":
            difficulty = prior
        elif experiment_name in ["cl_adaptive_raw", "cl_adaptive_laplace"]:
            difficulty = adaptive
        elif experiment_name == "cl_combined_staged":
            if progress <= staged_warmup:
                difficulty = prior
            elif progress <= staged_transition:
                alpha = (progress - staged_warmup) / (staged_transition - staged_warmup)
                difficulty = (1 - alpha) * prior + alpha * adaptive
            else:
                difficulty = 0.3 * prior + 0.7 * adaptive
        elif experiment_name == "cl_combined_optimized":
            alpha = 0.5 * (1 - math.cos(progress * math.pi)) 
            difficulty = (1 - alpha) * prior + alpha * adaptive
        else:
            # Default to prior if unknown
            difficulty = prior

        # Save Histogram
        import pandas as pd
        
        # Optimization: cl_prior is static, so only plot epoch 0
        should_plot_histogram = True
        if experiment_name == "cl_prior" and epoch > 0:
            should_plot_histogram = False
            
        if should_plot_histogram:
            df_csv = pd.read_csv(csv_train)
            results_root = os.path.dirname(os.path.dirname(metrics_log_path))
            save_per_class_histogram(
                difficulty, 
                df_csv["GT"].values, 
                mode=f"{dataset_name}_{experiment_name}_{strategy}_seed{seed}", 
                epoch=epoch,
                base_dir=results_root 
            )

        # --- Strategy Selection ---
        epoch_weights = None
        
        if strategy in ["reorder", "subsets", "weights"]:

            anti_mode = (cl_mode == "anti")
            
            indices, epoch_weights = get_strategy_batch(
                difficulty,
                epoch,
                epochs,
                strategy=strategy,
                anti_curriculum=anti_mode,
                seed=seed + epoch # vary seed per epoch for richness
            )
            
            subset = Subset(dataset_full, indices)

            shuffle_loader = False
            
        else:
            # Standard CL or Baseline Logic
            indices = get_curriculum_indices(
                difficulty, 
                epoch=epoch, 
                total_epochs=epochs, 
                pacing=pacing, 
                n_bins=n_bins,
                mode=cl_mode,
                hybrid_prob=hybrid_prob,
                sort_by_difficulty=ordered_cl
            )
            subset = Subset(dataset_full, indices)
            # Shuffle unless ordered_cl is True (Strict Sort)
            shuffle_loader = not ordered_cl
            
            
        if is_mil:
            subset_loader = DataLoader(subset, batch_size=1, shuffle=shuffle_loader, collate_fn=mil_collate)
        else:
            subset_loader = DataLoader(subset, batch_size=32, shuffle=shuffle_loader)
            
        # --- Train ---
        loss = train_epoch(model, subset_loader, optimizer, criterion, device, is_mil, sample_weights=epoch_weights)
        
        # --- Validate ---
        val_acc, val_f1, _, _, _ = evaluate_model(model, val_csv, folder, device)
        
        metrics_log.append({
             "epoch": epoch + 1,
             "loss": float(loss),
             "accuracy": float(val_acc),
             "f1": float(val_f1),
             "data_subset_size": len(subset)
         })

        if val_f1 > best_f1:
             best_f1 = val_f1
             best_state = model.state_dict()
             
        if scheduler:
            scheduler.step()
             
    # Save results
    save_metrics(metrics_log, metrics_log_path)
    if best_state:
        model.load_state_dict(best_state)
        torch.save(best_state, metrics_log_path.replace(".json", "_best.pt"))
        
    test_acc, test_f1, y_true, y_pred, y_prob = evaluate_model(model, test_csv, folder, device)
    save_metrics({"accuracy": float(test_acc), "f1": float(test_f1)}, test_metrics_path)
    
    ratios = [m["epoch"] / epochs for m in metrics_log]
    f1s = [m["f1"] for m in metrics_log]
    plot_performance_vs_ratio(ratios, f1s, "Validation F1", metrics_log_path.replace(".json", "_f1_plot.png"))

    return test_acc, test_f1

