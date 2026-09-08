
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import entropy

try:
    from torch_uncertainty.post_processing import LaplaceApprox
except ImportError:
    LaplaceApprox = None
    # print("[WARNING] Could not import LaplaceApprox. cl_adaptive_laplace will fail.")
except Exception as e:
    LaplaceApprox = None
    # print(f"[WARNING] Error importing LaplaceApprox: {e}")

from dataloader import AI4SKIN
from torch.utils.data import DataLoader

# ------------------------------------------------------------
# Utility: min-max normalize
# ------------------------------------------------------------
def _normalize(x):
    x = np.array(x, dtype=float)
    if x.size == 0:
        return x
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-8:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


# ------------------------------------------------------------
# 1. PRIOR DIFFICULTY
# ------------------------------------------------------------
def compute_prior_difficulty(csv_file):
    """
    Compute difficulty from annotator disagreement (entropy of markers).
    Returns: np.ndarray of shape (N_samples,)
    """
    df = pd.read_csv(csv_file)
    marker_cols = [f"Marker_{i}" for i in range(1, 11) if f"Marker_{i}" in df.columns]

    if not marker_cols:
        # Fallback to random or zero if no markers (e.g. inference)
        return np.zeros(len(df))

    markers = df[marker_cols].values.astype(float)

    prior = []
    for row in markers:
        valid = row[row != -1]
        if len(valid) <= 1:
            prior.append(0.0)
            continue

        _, counts = np.unique(valid, return_counts=True)
        p = counts / counts.sum()
        prior.append(entropy(p, base=2))

    return _normalize(prior)


# ------------------------------------------------------------
# MIL collate (for Laplace fitting if needed)
# ------------------------------------------------------------
def mil_collate(batch):
    xs = [item[0] for item in batch]   # list of bags, each (N_i, dim)
    ys = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return xs, ys


# ------------------------------------------------------------
# Helper: get logits for one sample (MIL or non-MIL)
# ------------------------------------------------------------
def _get_logits(model, x, device):
    """
    x:
      - MIL:  (N_patches, dim)
      - non-MIL (TITAN): (dim,)
    Returns:
      - logits: (C,)
    """
    if x.dim() == 2:      # MIL bag
        x = x.to(device)
        logits = model(x)        # expected (1, C)
        logits = logits.squeeze(0)
    else:                 # single embedding
        x = x.unsqueeze(0).to(device)   # (1, dim)
        logits = model(x)[0]           # (C,)
    return logits


# ------------------------------------------------------------
# 2. ADAPTIVE RAW DIFFICULTY (entropy of softmax)
# ------------------------------------------------------------
def compute_adaptive_difficulty(model, csv_file, folder, device):
    """
    Computes entropy-based difficulty (adaptive_raw).
    Works for both MIL and non-MIL.
    Returns: np.ndarray of shape (N_samples,)
    """
    model.eval()
    dataset = AI4SKIN(csv_file, folder)

    entropy_list = []

    with torch.no_grad():
        for x, _ in dataset:
            logits = _get_logits(model, x, device)     # (C,)
            probs = F.softmax(logits, dim=0)           # (C,)
            ent = -(probs * torch.log(probs + 1e-8)).sum().item()
            entropy_list.append(ent)

    return _normalize(entropy_list)



# ------------------------------------------------------------
# 5. ADAPTIVE LAPLACE DIFFICULTY (stub, requires LaplaceApprox wiring)
# ------------------------------------------------------------
def compute_adaptive_difficulty_laplace(
    model,
    csv_file,
    folder,
    device,
    num_samples=30
):
    """
    Laplace-based adaptive difficulty.
    Dataset interface unchanged: for x, _ in dataset
    Difficulty = mean predictive variance
    """

    model.eval()
    model.to(device)

    dataset = AI4SKIN(csv_file, folder)

    loader = DataLoader(dataset, batch_size=8, shuffle=False)


    if LaplaceApprox is None:
        raise ImportError("LaplaceApprox is not available. Please install torch-uncertainty[all].")

    # Laplace approximation
    laplace = LaplaceApprox(
        task="classification",
        model=model,
        weight_subset="last_layer",
        hessian_struct="diag",
    )

    try:
         laplace.fit(loader)
    except:
         # Fallback if fit fails (e.g. MIL batch handling)
         return np.zeros(len(dataset))

    difficulty_list = []

    with torch.no_grad():
        for x, _ in dataset:
            x = x.to(device)

            # Laplace posterior predictive
            preds = laplace.predictive_distribution(
                x.unsqueeze(0),     # (1, ...)
                num_samples=num_samples
            )
            # preds: (num_samples, 1, C)

            probs = F.softmax(preds, dim=-1)    # (S, 1, C)

            # Predictive variance
            var = probs.var(dim=0)               # (1, C)
            difficulty = var.mean().item()       # scalar

            difficulty_list.append(difficulty)

    return _normalize(difficulty_list)

# ------------------------------------------------------------
# 6. Curriculum pacing + binning
# ------------------------------------------------------------
def apply_pacing(ratio, pacing):
    if pacing == "linear":
        return ratio
    elif pacing == "exponential":
        return ratio ** 2
    elif pacing == "step":
        # New: 50% data immediately, 100% by half time
        return min(1.0, 0.5 + ratio)
    elif pacing == "inverse":
        return 1.0 - (1.0 - ratio) ** 2
    return ratio


def bin_difficulties(difficulty, n_bins=5):
    difficulty = np.array(difficulty)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(difficulty, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    return bin_idx, bins


def get_curriculum_indices(difficulty, epoch, total_epochs, pacing="linear", n_bins=5, mode="standard", hybrid_prob=0.0, sort_by_difficulty=False):
    """
    Returns indices of samples to train on this epoch.
    Options:
      - mode: 'standard' (easy first) or 'anti' (hard first)
      - hybrid_prob: float [0,1]. Probability of picking a random sample regardless of bin.
      - sort_by_difficulty: bool. If True, returns ALL indices sorted by difficulty (ignoring pacing/bins).
    """
    difficulty = np.array(difficulty)
    
    # --- ORDERED CURRICULUM MODE (Strict Sort) ---
    if sort_by_difficulty:
        sorted_indices = np.argsort(difficulty)
        if mode == "anti":
            return sorted_indices[::-1].tolist()
        else:
            return sorted_indices.tolist()

    # --- STANDARD SUBSET SAMPLING MODE ---
    bin_idx, _ = bin_difficulties(difficulty, n_bins)

    raw_ratio = (epoch + 1) / total_epochs
    ratio = apply_pacing(raw_ratio, pacing)

    total_samples = len(difficulty)
    total_to_sample = max(1, int(ratio * total_samples))

    # Define weights based on mode
    if mode == "anti":
        # Hard first: Higher weight to higher bins (indices n_bins-1)
        weights = np.array([(i + 1) for i in range(n_bins)], float)
    else:
        # Standard: Easy first (Linear Decay)
        weights = np.array([(n_bins - i) for i in range(n_bins)], float)
        
    weights /= weights.sum()

    selected = []
    
    # Pre-calculate counts per bin
    candidates_per_bin = [np.where(bin_idx == i)[0] for i in range(n_bins)]
    
    # Determine how many "pure curriculum" vs "hybrid random" samples
    n_random = int(total_to_sample * hybrid_prob)
    n_curriculum = total_to_sample - n_random
    
    # 1. Select Random subset (Hybrid)
    if n_random > 0:
        all_indices = np.arange(total_samples)
        random_chosen = np.random.choice(all_indices, n_random, replace=False)
        selected.extend(random_chosen.tolist())
        
    # 2. Select Curriculum subset
    if n_curriculum > 0:
        for i in range(n_bins):
            candidates = candidates_per_bin[i]
            if len(candidates) == 0:
                continue

            n_i = int(weights[i] * n_curriculum)
            
            # Simple heuristic
            if n_i <= 0 and weights[i] > 0 and n_curriculum > 0:
                 continue
                 
            # Sample
            chosen = np.random.choice(candidates, min(n_i, len(candidates)), replace=False)
            selected.extend(chosen.tolist())

    if not selected:  # fallback
        selected = list(range(total_samples))

    # Unique + sorted to remove duplicates from hybrid overlap
    return sorted(set(selected))


# ------------------------------------------------------------
# 7. NEW STRATEGIES (Reorder, Subsets, Weights)
# ------------------------------------------------------------
def get_strategy_batch(
    difficulty, 
    epoch, 
    total_epochs, 
    strategy="baseline", 
    anti_curriculum=False,
    seed=None
):
    """
    Implements strategies: 'reorder', 'subsets', 'weights' (from repo logic).
    Returns: 
       - indices: list of indices to use for this epoch
       - weights: optional weights per sample (for 'weights' strategy), matching indices order
    """
    if seed is not None:
        np.random.seed(seed)

    N = len(difficulty)
    indices = np.arange(N)
    probs = np.copy(difficulty)
    
    
    if anti_curriculum:
        # High difficulty = High probability
        # Normalize to sum to 1
        pass
    else:
        # Easy first -> Low difficulty = High probability
        probs = 1.0 - probs
        
    # Ensure raw probs are non-negative and sum to 1
    probs = np.clip(probs, 0, None)
    if probs.sum() == 0:
        probs = np.ones(N) / N
    else:
        probs = probs / probs.sum()

    # --- STRATEGIES ---
    
    if strategy == "reorder":
        # Weighted Shuffle (all samples)
        # Check non-zero entries
        non_zero_indices = np.where(probs > 0)[0]
        zero_indices = np.where(probs == 0)[0]
        
        if len(non_zero_indices) < N:
            p_non_zero = probs[non_zero_indices]
            if p_non_zero.sum() > 0:
                p_non_zero /= p_non_zero.sum()
            else:
                 p_non_zero = np.ones(len(non_zero_indices)) / len(non_zero_indices)

            part1 = np.random.choice(indices[non_zero_indices], size=len(non_zero_indices), replace=False, p=p_non_zero)
            part2 = np.random.choice(indices[zero_indices], size=len(zero_indices), replace=False)
            selected_indices = np.concatenate([part1, part2])
        else:
            selected_indices = np.random.choice(indices, size=N, replace=False, p=probs)
            
        return selected_indices.tolist(), None

    elif strategy == "subsets":
        # Grow subset size linearly: 25% -> 100%
        
        start_ratio = 0.25
        if total_epochs > 1:
            ratio = start_ratio + (1 - start_ratio) * (epoch / (total_epochs - 1))
        else:
            ratio = 1.0
        
        ratio = min(1.0, ratio)
        subset_size = int(ratio * N)
        subset_size = max(1, subset_size)
        
        # Check non-zero probs
        non_zero_indices = np.where(probs > 0)[0]
        
        if len(non_zero_indices) < subset_size:
            # Not enough non-zero prob items to fill the subset
            # Take all non-zero items first
            p_non_zero = probs[non_zero_indices] 
            if p_non_zero.sum() > 0:
                p_non_zero /= p_non_zero.sum()
            else:
                p_non_zero = np.ones(len(non_zero_indices)) / len(non_zero_indices)
                
            part1 = np.random.choice(indices[non_zero_indices], size=len(non_zero_indices), replace=False, p=p_non_zero)
            
            # Fill the rest with zero-prob items
            zero_indices = np.where(probs == 0)[0]
            remaining = subset_size - len(part1)
            # Cap remaining if necessary (shouldn't be if subset_size <= N)
            remaining = min(remaining, len(zero_indices))
            
            part2 = np.random.choice(indices[zero_indices], size=remaining, replace=False)
            selected_indices = np.concatenate([part1, part2])
        else:
            selected_indices = np.random.choice(indices, size=subset_size, replace=False, p=probs)
            
        return selected_indices.tolist(), None

    elif strategy == "weights":

        np.random.shuffle(indices)

        sample_weights = probs * N
        
        ordered_weights = sample_weights[indices] 
        return indices.tolist(), ordered_weights

    else:
        # Fallback / Baseline: Standard Shuffle
        np.random.shuffle(indices)
        return indices.tolist(), None
