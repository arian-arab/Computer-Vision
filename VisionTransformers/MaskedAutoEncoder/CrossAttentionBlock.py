import torch.nn as nn
    
class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim=64, heads=4, mlp_ratio=4.0):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim,
            heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )

        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, context):
        # Cross attention
        x = x + self.cross_attn(x, context, context)[0]
        x = self.norm1(x)

        # Feedforward
        x = x + self.mlp(x)
        x = self.norm2(x)

        return x
