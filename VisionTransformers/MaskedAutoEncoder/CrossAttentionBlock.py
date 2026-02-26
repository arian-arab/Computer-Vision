import torch.nn as nn

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim=64, heads=4, num_encoder_layers=2):
        super().__init__()

        # Transformer Encoder (runs first)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # Self-attention
        self.self_attn = nn.MultiheadAttention(embed_dim, heads, batch_first=True)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(embed_dim, heads, batch_first=True)

        # LayerNorms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, context):
        # 1️⃣ Encode first
        x = self.encoder(x)

        # 2️⃣ Self-attention
        x = x + self.self_attn(x, x, x)[0]
        x = self.norm1(x)

        # 3️⃣ Cross-attention
        x = x + self.cross_attn(x, context, context)[0]
        x = self.norm2(x)

        return x
