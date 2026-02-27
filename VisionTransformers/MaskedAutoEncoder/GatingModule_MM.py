import torch
import torch.nn as nn

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
        """
        x1, x2: (B, T, D) = (128, 8, 64)
        m1, m2: (B,) binary masks (0 = dropped, 1 = present)
        """

        B = x1.size(0)

        # --------------------------------------------------
        # 1️⃣ Zero out dropped modalities at feature level
        # --------------------------------------------------
        m1_feat = m1.view(B, 1, 1).float()
        m2_feat = m2.view(B, 1, 1).float()

        x1 = x1 * m1_feat
        x2 = x2 * m2_feat

        # --------------------------------------------------
        # 2️⃣ Global average pooling
        # --------------------------------------------------
        x1_g = x1.mean(dim=1)  # (B, D)
        x2_g = x2.mean(dim=1)  # (B, D)

        x = torch.cat([x1_g, x2_g], dim=1)  # (B, 2D)

        logits = self.mlp(x)  # (B, 2)

        # --------------------------------------------------
        # 3️⃣ Mask logits for missing modalities
        # --------------------------------------------------
        large_neg = -1e9

        logits[:, 0] = torch.where(m1 == 1, logits[:, 0], large_neg)
        logits[:, 1] = torch.where(m2 == 1, logits[:, 1], large_neg)

        alpha = torch.softmax(logits, dim=1)

        # --------------------------------------------------
        # 4️⃣ Entropy regularization
        # --------------------------------------------------
        eps = 1e-8
        entropy = - (alpha * torch.log(alpha + eps)).sum(dim=1).mean()

        # --------------------------------------------------
        # 5️⃣ Reshape gating weights for broadcasting
        # --------------------------------------------------
        alpha1 = alpha[:, 0].view(B, 1, 1)
        alpha2 = alpha[:, 1].view(B, 1, 1)

        return alpha1, alpha2, entropy
