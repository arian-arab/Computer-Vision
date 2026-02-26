import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=64, heads=4, depth=2, mlp_ratio=4.0):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

    def forward(self, x):
        return self.encoder(x)