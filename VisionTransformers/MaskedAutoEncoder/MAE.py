import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from PatchEmbed import PatchEmbed


class MAE(nn.Module):
    def __init__(self, img_size=28, patch_size=7, embed_dim=64, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # learnable tokens (placeholder) for maksed tokens, zero initialization for all masked tokens at the beginning of training

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        
        self.num_patches = self.patch_embed.num_patches

        # encoder
        self.encoder_pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # decoder
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        decoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=2)

        self.reconstruction_head = nn.Linear(embed_dim, patch_size * patch_size)

    def random_masking(self, x):
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))
    
        noise = torch.rand(B, N, device=x.device)
    
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
    
        # keep first subset
        ids_keep = ids_shuffle[:, :len_keep]
    
        # mask: 0 = keep, 1 = remove
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
    
        # masked input
        x_masked = torch.gather(
            x, dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )

        return x_masked, mask, ids_restore

    # ---------------------------------------------------
    # Encoder
    # ---------------------------------------------------             
    def encode(self, imgs):
        x = self.patch_embed(imgs)              # (B, N, D)
        x = x + self.encoder_pos_embed
        if self.training:
            x_masked, mask, ids_restore = self.random_masking(x)
        else:
            x_masked = x
            mask = None
            ids_restore = None   
            
        x_encoded = self.encoder(x_masked)      # batch_first=True
        return x_encoded, mask, ids_restore

    # ---------------------------------------------------
    # Decoder
    # ---------------------------------------------------
    def decode(self, x_encoded, ids_restore):        
        if self.training:
            B, N_visible, D = x_encoded.shape
            N_total = self.num_patches    
            # Create mask tokens
            mask_tokens = self.mask_token.repeat(B, N_total - N_visible, 1)    
            # Append mask tokens
            x_full = torch.cat([x_encoded, mask_tokens], dim=1)    
            # Restore original order
            x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D))
        else:
            x_full = x_encoded

        # Add decoder positional embedding
        x_full = x_full + self.decoder_pos_embed
        # Decode
        x_decoded = self.decoder(x_full)
        # Reconstruct patches
        pred = self.reconstruction_head(x_decoded)
        return pred

    # ---------------------------------------------------
    # Forward
    # ---------------------------------------------------    
    def forward(self, imgs):
        x_encoded, mask, ids_restore = self.encode(imgs)
        pred = self.decode(x_encoded, ids_restore)
        return pred, x_encoded, mask
    
# ------------------------
# Load Datasets
# ------------------------
from load_train_test_datasets import load_train_test_datasets
loader_train, loader_test = load_train_test_datasets()

# ------------------------
# Model Training
# ------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model = MAE(embed_dim=128, mask_ratio=0.75).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


n_epochs = 100
loss_vals = []
for epoch in range(n_epochs):
    model.train()
   
    total_loss = 0
    for imgs, _, _, _ in loader_train:
        imgs = imgs.to(device)

        # pred, _ = model(imgs)
        pred, z, mask = model(imgs)

        # target patches
        patches = imgs.unfold(2,7,7).unfold(3,7,7)
        patches = patches.contiguous().view(imgs.size(0), 16, -1)
        
        loss = (pred - patches) ** 2
        loss = loss.mean(dim=-1)          # mean over feature dimension
        loss = (loss * mask).sum() / mask.sum()

        loss = F.mse_loss(pred, patches)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch +1}, Loss {total_loss/len(loader_train)}")
    loss_vals.append(total_loss)
    
    
plt.figure()
plt.plot(np.arange(1, n_epochs+1), loss_vals)
plt.xlabel('Epochs')
plt.ylabel('Total Loss')
plt.show()

torch.save(model.state_dict(), 'MAE.pth')

model.load_state_dict(torch.load('MAE.pth'))
model.eval()
    
# ------------------------
# Model Inference
# ------------------------
def patches_to_image(patches):
    B = patches.shape[0]
    patches = patches.view(B, 4, 4, 7, 7)
    patches = patches.permute(0,1,3,2,4)
    images = patches.reshape(B, 1, 28, 28)
    return images

def visualize_reconstruction(model, loader, device):
    model.eval()

    img, _, _, _ = next(iter(loader))
    img = img.to(device)

    with torch.no_grad():
        pred, _, _ = model(img)

    recon = patches_to_image(pred.cpu())
    
    img = img.cpu()
    num_show = 12
    fig, axes = plt.subplots(2, num_show, figsize=(8, 3))
    for i in range(num_show):
        axes[0,i].imshow(img[i,0], cmap='gray')        
        axes[0,i].axis('off')

        axes[1,i].imshow(recon[i,0], cmap='gray')        
        axes[1,i].axis('off')
    plt.tight_layout()
    plt.show()

visualize_reconstruction(model, loader_test, device)
