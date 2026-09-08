"""Cumulative agreement curriculum on MHIST.

Reproduces the staged curriculum of "Learn like a Pathologist: Curriculum
Learning by Annotator Agreement for Histopathology Image Classification" on
frozen foundation-model embeddings rather than a trained ResNet, so that the
pacing mechanism is isolated from the backbone.

The paper states 50 epochs but does not make clear whether that is per stage or
across the whole sequence, so both readings are supported and matched
against a non-CL baseline given the identical total number of epochs:

    --budget total     T epochs split as evenly as possible over 4 stages
                       (T=50 -> 13/13/12/12); baseline also gets T
    --budget perstage  T epochs in EACH stage (4T total);
                       baseline gets 4T

Checkpoint selection uses validation macro-F1 over the whole run (never test),
and the selected checkpoint is scored once on test.

    python run_wei_curriculum.py --budget total --epochs 50
    python run_wei_curriculum.py --budget perstage --epochs 50
"""

import argparse, json, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)

from models import build_model                                    # noqa: E402
from run_baselines import EmbeddingDataset, collate_mil           # noqa: E402
from test import evaluate_model                                   # noqa: E402
from utils import detect_input_dim_and_mil, set_seed              # noqa: E402

AGREEMENT_STAGES = [7, 6, 5, 4]     # cumulative: >=7/7, >=6/7, >=5/7, >=4/7


def stage_indices(csv_train):
    """Cumulative index sets, easiest (most unanimous) stage first."""
    df = pd.read_csv(csv_train)
    agree = np.maximum(df.n_ssa.values, 7 - df.n_ssa.values)
    return [np.where(agree >= lvl)[0] for lvl in AGREEMENT_STAGES]


def split_budget(total, n):
    """Split T epochs as evenly as possible over n stages (50 -> 13,13,12,12)."""
    base, rem = divmod(total, n)
    return [base + (1 if i < rem else 0) for i in range(n)]


def train(model, ds, idx, epochs, opt, sched, crit, device, is_mil,
          csv_val, folder, state, log):
    """Train on `idx` for `epochs`, tracking the global best-validation state."""
    bs = 1 if is_mil else 32
    loader = DataLoader(Subset(ds, list(idx)), batch_size=bs, shuffle=True,
                        collate_fn=collate_mil if is_mil else None)
    for _ in range(epochs):
        model.train()
        for x, y, _s, _i in loader:
            x = x[0].to(device) if is_mil else x.to(device)
            y = y.to(device)
            opt.zero_grad()
            out = model(x)
            if out.dim() == 1:
                out = out.unsqueeze(0)
            loss = crit(out, y)
            loss.backward(); opt.step()
        _, vf1, *_ = evaluate_model(model, csv_val, folder, device)
        log.append({"epoch": len(log) + 1, "n_train": len(idx), "val_f1": float(vf1)})
        if vf1 > state["best"]:
            state["best"] = vf1
            state["sd"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if sched:
            sched.step()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoders", nargs="+", default=["CONCHv1.5", "UNI2", "VIRCHOW2"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 53, 78, 102, 294])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--budget", choices=["total", "perstage"], default="total")
    p.add_argument("--csv-dir", default=os.path.join(HERE, "..", "csv_mhist"))
    p.add_argument("--data-root", default=os.path.join(HERE, "..", "..", "data_mhist"))
    p.add_argument("--results-root", default=None)
    a = p.parse_args()

    c_tr, c_va, c_te = (os.path.join(a.csv_dir, f"{s}.csv") for s in ("train", "val", "test"))
    stages = stage_indices(c_tr)
    if a.budget == "total":
        per_stage = split_budget(a.epochs, len(stages)); total = a.epochs
    else:
        per_stage = [a.epochs] * len(stages); total = a.epochs * len(stages)

    out = a.results_root or os.path.join(HERE, "..", "..", f"results_wei_mhist_{a.budget}")
    logs = os.path.join(out, "logs"); os.makedirs(logs, exist_ok=True)
    print(f"[PLAN] budget={a.budget}  stages={[len(s) for s in stages]}  "
          f"epochs/stage={per_stage}  TOTAL={total} (baseline gets {total})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for enc in a.encoders:
        folder = os.path.join(a.data_root, enc)
        dim, is_mil = detect_input_dim_and_mil(c_tr, folder)
        if dim == 0:
            print(f"[SKIP] {enc}"); continue
        ds = EmbeddingDataset(c_tr, folder)
        for method in ["wei_curriculum", "non_cl"]:
            for seed in a.seeds:
                tag = f"{method}_dataset-{enc}_seed-{seed}_{a.budget}"
                lj, tj = os.path.join(logs, tag + ".json"), os.path.join(logs, tag + "_test.json")
                if os.path.exists(tj):
                    continue
                print(f"--> {tag}", flush=True)
                set_seed(seed)
                model = build_model(dim, 2, is_mil).to(device)
                opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
                sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
                crit = nn.CrossEntropyLoss()
                state, log = {"best": 0.0, "sd": None}, []

                if method == "wei_curriculum":
                    for idx, ep in zip(stages, per_stage):     # sequential fine-tuning
                        train(model, ds, idx, ep, opt, sched, crit, device, is_mil,
                              c_va, folder, state, log)
                else:
                    train(model, ds, np.arange(len(ds)), total, opt, sched, crit,
                          device, is_mil, c_va, folder, state, log)

                if state["sd"]:
                    model.load_state_dict(state["sd"])
                acc, f1, *_ = evaluate_model(model, c_te, folder, device)
                json.dump(log, open(lj, "w"), indent=2)
                json.dump({"accuracy": float(acc), "f1": float(f1),
                           "best_val_f1": float(state["best"]), "total_epochs": total},
                          open(tj, "w"), indent=2)
                print(f"      test acc={acc:.4f} f1={f1:.4f} (best val {state['best']:.4f})")
    print("[DONE]")


if __name__ == "__main__":
    main()
