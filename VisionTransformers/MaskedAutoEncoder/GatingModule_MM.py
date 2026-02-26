import torch.nn as nn
import torch

class GatingModule_MM(nn.Module):
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

    def forward(self, x1, x2, m1, m2):
        # Global average pool
        x1_g = x1.mean(dim=1)
        x2_g = x2.mean(dim=1)

        x = torch.cat([x1_g, x2_g], dim=1)
        logits = self.mlp(x)

        # Force missing modality logits to -inf
        minus_inf = torch.finfo(logits.dtype).min
        logits[:, 0] = torch.where(m1 == 1, logits[:, 0], minus_inf)
        logits[:, 1] = torch.where(m2 == 1, logits[:, 1], minus_inf)

        alpha = torch.softmax(logits, dim=1)

        eps = 1e-8
        entropy = - (alpha * torch.log(alpha + eps)).sum(dim=1).mean()

        alpha1 = alpha[:, 0].unsqueeze(-1).unsqueeze(-1)
        alpha2 = alpha[:, 1].unsqueeze(-1).unsqueeze(-1)

        return alpha1, alpha2, entropy