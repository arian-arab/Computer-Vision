import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_chans=1, embed_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                    # (B, D, H, W)
        x = x.flatten(2).transpose(1, 2)    # (B, N, D)
        return x