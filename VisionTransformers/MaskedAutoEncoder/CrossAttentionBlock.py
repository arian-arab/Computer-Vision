import torch.nn as nn

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim=64, heads=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, heads, batch_first=True)        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, context):
        # Self-attention + residual + norm
        x = x + self.self_attn(x, x, x)[0]
        x = self.norm1(x)
        # Cross-attention + residual + norm
        x = x + self.cross_attn(x, context, context)[0]
        x = self.norm2(x)
        return x