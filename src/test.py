import numpy as np
import pandas as pd
import torch
import os
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

from dataloader import load_embedding, resolve_embedding_path


def evaluate_model(model, csv_file, folder, device):
    """
    Evalúa el modelo sobre el CSV indicado (val o test).
    Usa balanced accuracy y F1 macro.
    """
    df = pd.read_csv(csv_file)

    all_preds = []
    all_labels = []
    y_prob = []

    model.eval()
    with torch.no_grad():
        for _, row in df.iterrows():

            key = 'ano' if 'ano' in row else 'WSI'
            base_filename = str(row[key])

            file_path, file_type = resolve_embedding_path(
                folder, base_filename, allow_prefix_strip=(key == 'WSI')
            )
            if file_path is None:
                raise FileNotFoundError(
                    f"Could not find .npy or .pt file for {base_filename} in {folder}")

            emb = load_embedding(file_path, file_type)
            if not isinstance(emb, torch.Tensor):
                x = torch.tensor(emb, dtype=torch.float32).to(device)
            else:
                x = emb.clone().detach().float().to(device)

            # TITAN -> single embedding
            if x.dim() == 1:
                x = x.unsqueeze(0)        # shape (1, dim)
            # MIL -> bag representation (N_patches, dim)

            logits = model(x)
            probs = torch.softmax(logits, dim=1) if logits.dim() == 2 else torch.softmax(logits, dim=0)

            if probs.dim() == 1:
                # (num_classes,) → MIL single sample
                pred = probs.argmax().item()
            else:
                # (1, num_classes)
                pred = probs.argmax(dim=1).item()

            all_preds.append(pred)
            all_labels.append(int(row["GT"]))
            y_prob.append(probs.squeeze(0).cpu().numpy())

    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    std_acc = accuracy_score(all_labels, all_preds)

    print(f"Accuracy (standard): {std_acc:.4f}")
    print(f"Balanced accuracy:   {bal_acc:.4f}")
    print(f"F1 macro:            {macro_f1:.4f}")

    return bal_acc, macro_f1, all_labels, all_preds, y_prob
