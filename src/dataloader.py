from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import torch

def resolve_embedding_path(folder, base_filename, allow_prefix_strip=False):
    """Locate an embedding file, accepting either .npy or .pt.

    Externally prepared datasets are written as .npy while the AI4SKIN copies
    under src/processed are .pt, so both callers (AI4SKIN and evaluate_model)
    have to cope with either. Returns (path, kind) or (None, None).

    `allow_prefix_strip` retries without a leading "AI4SKIN_", which some of the
    AI4SKIN folders need when keyed by WSI.
    """
    candidates = [(base_filename + '.npy', 'npy'), (base_filename + '.pt', 'pt')]
    if allow_prefix_strip and base_filename.startswith("AI4SKIN_"):
        stripped = base_filename.replace("AI4SKIN_", "", 1)
        candidates += [(stripped + '.npy', 'npy'), (stripped + '.pt', 'pt')]

    for name, kind in candidates:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path, kind
    return None, None


def load_embedding(path, kind):
    """Read one embedding file into a numpy array or tensor."""
    if kind == 'npy':
        return np.load(path, allow_pickle=True)
    embedding = torch.load(path, map_location='cpu')
    return embedding.numpy() if isinstance(embedding, torch.Tensor) else embedding


class AI4SKIN(Dataset):
    def __init__(self, csv_file, folder, difficulty_scores=None, threshold=1.0):
        self.data = pd.read_csv(csv_file)
        self.folder = folder
        self.threshold = threshold
    
        if difficulty_scores is not None:
            # Map difficulties only if provided
            self.data['difficulty'] = self.data['ano'].map(dict(zip(self.data['ano'], difficulty_scores)))
            self._filter_easy_samples()
        else:
            # If no difficulties, load full dataset (no filtering)
            self.data['difficulty'] = 0  # or you can skip this column entirely

    def _filter_easy_samples(self):
        self.data = self.data[self.data['difficulty'] <= self.threshold].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Use 'ano' column for filename if available, fallback to WSI
        if 'ano' in row:
            key = 'ano'
        else:
            key = 'WSI'
            
        base_filename = str(row[key])
        file_path, file_type = resolve_embedding_path(
            self.folder, base_filename, allow_prefix_strip=(key == 'WSI')
        )

        if file_path is None:
            raise FileNotFoundError(f"Could not find .npy or .pt file for {base_filename} in {self.folder}")

        try:
            embedding = load_embedding(file_path, file_type)
        except Exception as e:
            raise FileNotFoundError(f"Could not load {file_path} (Original: {base_filename}). Error: {e}")

        label = int(row['GT'])  # ensure label is int

        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding, dtype=torch.float32)
        else:
            embedding = embedding.clone().detach().float()
        return embedding, torch.tensor(label, dtype=torch.long)
