import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        decoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=2)
        self.head = nn.Linear(embed_dim, 7*7)

    def forward(self, z):
        x = self.decoder(z)
        patches = self.head(x)
        return patches