import torch.nn as nn
import torch

class GatingModule(nn.Module):
    def __init__(self, embed_dim=64, hidden_dim=128, dropout=0.1):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x1, x2):
        # Global Average Pool
        x1_g = x1.mean(dim=1)   # (B, D)
        x2_g = x2.mean(dim=1)

        x = torch.cat([x1_g, x2_g], dim=1)

        logits = self.mlp(x)
        alpha = torch.softmax(logits, dim=1)

        # ---- Entropy Regularization ----
        eps = 1e-8  # numerical stability
        entropy = - (alpha * torch.log(alpha + eps)).sum(dim=1).mean()

        alpha1 = alpha[:, 0].unsqueeze(-1).unsqueeze(-1)
        alpha2 = alpha[:, 1].unsqueeze(-1).unsqueeze(-1)
        return alpha1, alpha2, entropy