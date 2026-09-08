# models.py
import torch
from torch import nn

class SimpleMLP_Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim=64, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU()
        )
        self.embed_dim = embed_dim

    def forward(self, x):
        # x is (B, input_dim) or (input_dim,) if batch_size=1
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)  # (B, embed_dim)


class TransMIL_Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=4, num_layers=2):
        super().__init__()

        self.project = nn.Linear(input_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.embed_dim = embed_dim

    def forward(self, x):
        """
        x: (N_patches, input_dim) for a single bag
        returns: (1, embed_dim)
        """
        if x.dim() != 2:
            # Fallback or error if not 2D
             if x.dim() == 3 and x.size(0) == 1: # (1, N, D)
                x = x.squeeze(0)
             else:
                raise ValueError(f"TransMIL_Encoder expects (N_patches, input_dim), got {x.shape}")

        H = self.project(x).unsqueeze(0)  # (1, N, D)
        H = self.transformer(H)           # (1, N, D)
        bag_emb = H.mean(dim=1)          # (1, D)
        return bag_emb


class ClassifierHead(nn.Module):
    def __init__(self, in_dim=64, num_classes=6):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        logits = self.fc(x)
        # make sure we always return (B, C)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        return logits


def build_model(input_dim: int, num_classes: int, is_mil: bool) -> nn.Module:
    """
    Builds the model based on input dimension and MIL status.
    """
    if not is_mil:
        encoder = SimpleMLP_Encoder(input_dim=input_dim, embed_dim=64)
    else:
        encoder = TransMIL_Encoder(input_dim=input_dim, embed_dim=64)

    classifier = ClassifierHead(in_dim=64, num_classes=num_classes)
    model = nn.Sequential(encoder, classifier)
    return model
