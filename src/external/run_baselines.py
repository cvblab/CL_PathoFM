"""Label-noise-robust and annotator-aware baselines.

Implemented:

  non_cl               plain cross-entropy (reference; also validates this loop)
  label_smoothing      CE with eps=0.1                              
  gce                  Generalized Cross Entropy, q=0.7             
  focal                Focal loss, gamma=2                          
  class_balanced       CE weighted by effective number, beta=0.999  
  soft_labels          KL to the annotator vote distribution        
  random_curriculum    subsets pacing driven by RANDOM difficulty   
  reverse_curriculum   subsets pacing, hardest-first                
  agreement_weighting  static per-sample weight = annotator agreement 


Mirrors `run_external.py`'s `non_cl` path exactly so numbers are comparable:
non-MIL uses AdamW(lr=1e-4, wd=1e-4) + CosineAnnealingLR and batch 32; MIL uses
Adam(lr=1e-3, wd=1e-4), no scheduler, batch 1. Best epoch is chosen on
validation macro-F1, then that checkpoint is scored on test. Output filenames
and JSON shapes match results_*/logs/ so the existing aggregation works.

Usage
-----
    python run_baselines.py --list
    python run_baselines.py --dry-run                    # cost estimate only
    python run_baselines.py --datasets mhist --methods label_smoothing gce
    python run_baselines.py --datasets mhist crowdgleason ai4skin
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from curriculum import compute_prior_difficulty, get_strategy_batch  # noqa: E402
from dataloader import load_embedding, resolve_embedding_path        # noqa: E402
from models import build_model                                       # noqa: E402
from test import evaluate_model                                      # noqa: E402
from utils import detect_input_dim_and_mil, set_seed                 # noqa: E402

SEEDS = [42, 53, 78, 102, 294]
N_MARKERS = 10

DATASETS = {
    "ai4skin":      ("csv",               "processed",                6),
    "mhist":        ("csv_mhist",         "../data_mhist",            2),
    "crowdgleason": ("csv_crowdgleason",  "/root/data_crowdgleason",  4),
}


METHODS = [
    "non_cl",
    "random_curriculum", "reverse_curriculum", "agreement_weighting",
    "label_smoothing", "gce", "soft_labels",
    "focal", "class_balanced",
    "prior_curriculum_subsets", "prior_curriculum_weights",
    "random_curriculum_weights", "reverse_curriculum_weights",
]

CL_SCHEDULE = {
    "random_curriculum": "subsets", "reverse_curriculum": "subsets",
    "prior_curriculum_subsets": "subsets",
    "prior_curriculum_weights": "weights",
    "random_curriculum_weights": "weights",
    "reverse_curriculum_weights": "weights",
}


# ---------------------------------------------------------------- losses ----
class FocalLoss(nn.Module):
    """Down-weights easy examples so training focuses on hard ones (gamma=2)."""

    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma, self.weight = gamma, weight

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=-1)
        logpt = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        loss = -((1 - logpt.exp()) ** self.gamma) * logpt
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()


class GCELoss(nn.Module):
    """Generalized Cross Entropy (Zhang & Sabuncu 2018): (1 - p_y^q) / q.

    Interpolates between MAE (q->1, noise-robust but slow) and CE (q->0). The
    standard q=0.7 is the noise-robustness baseline R1 asked for.
    """

    def __init__(self, q=0.7):
        super().__init__()
        self.q = q

    def forward(self, logits, target):
        p = F.softmax(logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
        p = p.clamp(min=1e-7)
        return ((1.0 - p ** self.q) / self.q).mean()


class SoftLabelLoss(nn.Module):
    """KL(annotator vote distribution || prediction).

    Trains on the full crowd distribution instead of a single aggregated label,
    so a 4-vs-3 split is not treated as though it were unanimous.
    """

    def forward(self, logits, soft_target):
        return F.kl_div(F.log_softmax(logits, dim=-1), soft_target,
                        reduction="batchmean")


def class_balanced_weights(labels, num_classes, beta=0.999, device="cpu"):
    """Effective-number reweighting (Cui et al. 2019)."""
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(float)
    counts = np.maximum(counts, 1.0)
    eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    w = 1.0 / eff
    w = w / w.sum() * num_classes
    return torch.tensor(w, dtype=torch.float32, device=device)


# --------------------------------------------------------------- targets ----
def soft_targets_from_markers(csv_file, num_classes):
    """Per-sample distribution over classes from the Marker_* annotator columns.

    Rows with no usable annotations fall back to a one-hot on GT, so the loss is
    always well-defined.
    """
    df = pd.read_csv(csv_file)
    cols = [f"Marker_{i}" for i in range(1, N_MARKERS + 1) if f"Marker_{i}" in df.columns]
    out = np.zeros((len(df), num_classes), dtype=np.float32)
    n_fallback = 0
    for i, row in enumerate(df.itertuples(index=False)):
        votes = []
        if cols:
            raw = np.array([getattr(row, c) for c in cols], dtype=float)
            votes = [int(v) for v in raw[raw != -1] if 0 <= int(v) < num_classes]
        if votes:
            for v in votes:
                out[i, v] += 1.0
            out[i] /= out[i].sum()
        else:
            out[i, int(getattr(row, "GT"))] = 1.0
            n_fallback += 1
    return out, n_fallback


def agreement_weights(csv_file):
    """Static per-sample weight = 1 - normalised annotator entropy, mean-1 scaled.

    High agreement -> high weight. This is the no-curriculum version of the
    paper's prior signal.
    """
    prior = np.asarray(compute_prior_difficulty(csv_file), dtype=float)
    w = 1.0 - prior
    m = w.mean()
    return w / m if m > 1e-8 else np.ones_like(w)


# --------------------------------------------------------------- dataset ----
class EmbeddingDataset(Dataset):
    """Same resolution logic as dataloader.AI4SKIN, plus soft target and index."""

    def __init__(self, csv_file, folder, soft=None):
        self.data = pd.read_csv(csv_file)
        self.folder = folder
        self.soft = soft

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        key = "ano" if "ano" in row else "WSI"
        path, kind = resolve_embedding_path(self.folder, str(row[key]),
                                            allow_prefix_strip=(key == "WSI"))
        if path is None:
            raise FileNotFoundError(f"no embedding for {row[key]} in {self.folder}")
        emb = load_embedding(path, kind)
        x = (torch.tensor(emb, dtype=torch.float32)
             if not isinstance(emb, torch.Tensor) else emb.clone().detach().float())
        y = torch.tensor(int(row["GT"]), dtype=torch.long)
        s = (torch.tensor(self.soft[idx]) if self.soft is not None
             else torch.zeros(1))
        return x, y, s, idx


def collate_mil(batch):
    return ([b[0] for b in batch],
            torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]),
            [b[3] for b in batch])


# ----------------------------------------------------------------- train ----
def run_one(method, csv_train, csv_val, csv_test, folder, input_dim, is_mil,
            num_classes, seed, epochs, device, log_json, test_json):
    """One (method, encoder, seed) run. Mirrors run_external.py's non_cl path."""
    set_seed(seed)

    soft = None
    if method == "soft_labels":
        soft, n_fb = soft_targets_from_markers(csv_train, num_classes)
        if n_fb:
            print(f"      [soft_labels] {n_fb} rows fell back to one-hot GT")

    ds = EmbeddingDataset(csv_train, folder, soft=soft)
    labels = pd.read_csv(csv_train)["GT"].astype(int).values

    model = build_model(input_dim, num_classes, is_mil).to(device)
    if not is_mil:
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = None

    # --- loss selection -----------------------------------------------------
    soft_loss = SoftLabelLoss()
    if method == "label_smoothing":
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    elif method == "gce":
        criterion = GCELoss(q=0.7)
    elif method == "focal":
        criterion = FocalLoss(gamma=2.0)
    elif method == "class_balanced":
        criterion = nn.CrossEntropyLoss(
            weight=class_balanced_weights(labels, num_classes, device=device))
    elif method == "agreement_weighting":
        criterion = nn.CrossEntropyLoss(reduction="none")
    else:
        criterion = nn.CrossEntropyLoss()

    sample_w = (torch.tensor(agreement_weights(csv_train), dtype=torch.float32, device=device)
                if method == "agreement_weighting" else None)
    criterion_none = nn.CrossEntropyLoss(reduction="none")

    # --- curriculum-control difficulty --------------------------------------
    difficulty = None
    if method.startswith("random_curriculum"):
        difficulty = np.random.RandomState(seed).rand(len(ds))
    elif method.startswith(("reverse_curriculum", "prior_curriculum")):
        difficulty = np.asarray(compute_prior_difficulty(csv_train), dtype=float)

    bs = 1 if is_mil else 32
    collate = collate_mil if is_mil else None

    best_f1, best_state, log = 0.0, None, []
    for epoch in range(epochs):
        # The two curriculum controls reuse the SAME sampler as the real
        # cl_prior+subsets runs; only `difficulty` differs.
        cl_w = None
        if difficulty is not None:
            idxs, wts = get_strategy_batch(
                difficulty, epoch, epochs, strategy=CL_SCHEDULE[method],
                anti_curriculum=method.startswith("reverse_curriculum"), seed=seed)
            sub = torch.utils.data.Subset(ds, list(idxs))
            loader = DataLoader(sub, batch_size=bs, shuffle=True, collate_fn=collate)
            # For the "weights" schedule the difficulty enters ONLY through these
            # per-sample loss weights; discarding them makes the run a no-op.
            if wts is not None:
                m = np.ones(len(ds), dtype=np.float32)
                m[np.asarray(idxs, dtype=int)] = np.asarray(wts, dtype=np.float32)
                cl_w = torch.tensor(m, device=device)
        else:
            loader = DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=collate)

        model.train()
        total = 0.0
        for x, y, s, idx in loader:
            x = x[0].to(device) if is_mil else x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)

            if method == "soft_labels":
                loss = soft_loss(logits, s.to(device))
            elif method == "agreement_weighting":
                per = criterion(logits, y)
                loss = (per * sample_w[torch.tensor(idx, device=device)]).mean()
            elif cl_w is not None:
                per = criterion_none(logits, y)
                loss = (per * cl_w[torch.tensor(idx, device=device)]).mean()
            else:
                loss = criterion(logits, y)

            loss.backward()
            optimizer.step()
            total += float(loss.item())

        val_acc, val_f1, *_ = evaluate_model(model, csv_val, folder, device)
        log.append({"epoch": epoch + 1, "loss": total / max(1, len(loader)),
                    "accuracy": float(val_acc), "f1": float(val_f1)})
        if val_f1 > best_f1:
            best_f1, best_state = val_f1, {k: v.detach().cpu().clone()
                                           for k, v in model.state_dict().items()}
        if scheduler:
            scheduler.step()

    with open(log_json, "w") as f:
        json.dump(log, f, indent=2)
    if best_state:
        model.load_state_dict(best_state)
        torch.save(best_state, log_json.replace(".json", "_best.pt"))

    test_acc, test_f1, *_ = evaluate_model(model, csv_test, folder, device)
    with open(test_json, "w") as f:
        json.dump({"accuracy": float(test_acc), "f1": float(test_f1)}, f, indent=2)
    return test_acc, test_f1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=["mhist"], choices=list(DATASETS))
    p.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    p.add_argument("--encoders", nargs="*", default=["UNI2", "VIRCHOW2", "CONCHv1.5"])
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--results-root", default=None,
                   help="Default: ../../results_baselines_<dataset>")
    p.add_argument("--min-per-run", type=float, default=None,
                   help="Minutes per run, for the cost estimate only")
    p.add_argument("--dry-run", action="store_true", help="Plan and exit")
    p.add_argument("--list", action="store_true", help="List methods and exit")
    args = p.parse_args()

    if args.list:
        print("methods:")
        for m in METHODS:
            print(f"  {m}")
        print("\ndatasets:")
        for d, (c, r, n) in DATASETS.items():
            print(f"  {d:14} csv={c:20} data={r:26} classes={n}")
        return

    src = os.path.join(HERE, "..")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plan = []

    for tag in args.datasets:
        csv_dir, data_root, ncls = DATASETS[tag]
        csv_dir = csv_dir if os.path.isabs(csv_dir) else os.path.join(src, csv_dir)
        data_root = data_root if os.path.isabs(data_root) else os.path.join(src, data_root)
        c_tr = os.path.join(csv_dir, "train.csv")
        c_va = os.path.join(csv_dir, "val.csv")
        c_te = os.path.join(csv_dir, "test.csv")
        if not os.path.exists(c_tr):
            print(f"[SKIP] {tag}: no {c_tr}")
            continue

        results = args.results_root or os.path.join(src, "..", f"results_baselines_{tag}")
        logs = os.path.join(results, "logs")
        os.makedirs(logs, exist_ok=True)

        encs = [e for e in args.encoders if os.path.isdir(os.path.join(data_root, e))]
        if not encs:
            print(f"[SKIP] {tag}: none of {args.encoders} under {data_root}")
            continue

        for enc in encs:
            folder = os.path.join(data_root, enc)
            dim, is_mil = detect_input_dim_and_mil(c_tr, folder)
            if dim == 0:
                print(f"[SKIP] {tag}/{enc}: unreadable embeddings")
                continue
            for method in args.methods:
                for seed in args.seeds:
                    name = f"{method}_dataset-{enc}_seed-{seed}_baseline"
                    lj = os.path.join(logs, f"{name}.json")
                    tj = os.path.join(logs, f"{name}_test.json")
                    if os.path.exists(tj):
                        continue
                    plan.append(dict(tag=tag, method=method, enc=enc, seed=seed,
                                     folder=folder, dim=dim, is_mil=is_mil,
                                     ncls=ncls, c_tr=c_tr, c_va=c_va, c_te=c_te,
                                     lj=lj, tj=tj, results=results))

    print(f"\n[PLAN] {len(plan)} runs to do "
          f"(datasets={args.datasets}, {len(args.methods)} methods, "
          f"{len(args.seeds)} seeds, {args.epochs} epochs)")
    for tag in args.datasets:
        n = sum(1 for x in plan if x["tag"] == tag)
        if n:
            print(f"         {tag}: {n}")
    if args.min_per_run:
        print(f"[PLAN] estimated {len(plan) * args.min_per_run / 60:.1f} h "
              f"at {args.min_per_run} min/run")
    if args.dry_run or not plan:
        return

    for i, job in enumerate(plan, 1):
        print(f"\n[{i}/{len(plan)}] {job['tag']} | {job['method']} | "
              f"{job['enc']} | seed {job['seed']}", flush=True)
        try:
            acc, f1 = run_one(job["method"], job["c_tr"], job["c_va"], job["c_te"],
                              job["folder"], job["dim"], job["is_mil"], job["ncls"],
                              job["seed"], args.epochs, device, job["lj"], job["tj"])
            print(f"      test acc={acc:.4f} f1={f1:.4f}")
        except Exception as exc:
            import traceback
            print(f"[ERROR] {job['method']}/{job['enc']}/{job['seed']}: {exc}")
            traceback.print_exc()

    print("\n[DONE]")


if __name__ == "__main__":
    main()
