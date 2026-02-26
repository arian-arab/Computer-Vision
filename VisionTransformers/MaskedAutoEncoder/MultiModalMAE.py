import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from PatchEmbed import PatchEmbed
from CrossAttentionBlock import CrossAttentionBlock
from GatingModule_MM import GatingModule_MM
from Decoder import Decoder    
from TransformerEncoder import TransformerEncoder

class MultiModalMAE_MM(nn.Module):
    def __init__(self, embed_dim=64, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.patch_embed = PatchEmbed(embed_dim=embed_dim)
        
        num_patches = self.patch_embed.num_patches

        self.encoder_pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        
        self.encoder1 = TransformerEncoder(embed_dim)
        self.encoder2 = TransformerEncoder(embed_dim)
        
        self.cross_block = CrossAttentionBlock(embed_dim)

        self.gating = GatingModule_MM(embed_dim)
        
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))

        self.decoder_1 = Decoder(embed_dim)
        self.decoder_2 = Decoder(embed_dim)        
    
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
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        return x_masked, mask, ids_restore, ids_keep   
    
    def encoder(self, img1, img2, m1=None, m2=None):
        B = img1.size(0)
        device = img1.device    
        
        if self.training:
            probs = torch.rand(B, 1, device=device)
    
            m1 = torch.zeros(B,1, device=device)
            m2 = torch.zeros(B,1, device=device)
            
            m1[probs >= 0.33] = 1
            m2[probs < 0.66] = 1
        else:
            m1 = m1.view(B,1).to(device)
            m2 = m2.view(B,1).to(device)             
            
        x1 = self.patch_embed(img1) + self.encoder_pos_embed
        x2 = self.patch_embed(img2) + self.encoder_pos_embed
        
        if self.training:
            x1_masked, mask, ids_restore, ids_keep = self.random_masking(x1)
            x2_masked = torch.gather(x2, 1, ids_keep.unsqueeze(-1).repeat(1,1,x2.shape[-1]))
        else:
            x1_masked = x1
            x2_masked = x2
            ids_restore = None            
            ids_keep = None
            mask = None              
        
        # Now apply self encoders
        x1_masked = self.encoder1(x1_masked)
        x2_masked = self.encoder2(x2_masked)             
        
        # Clone to preserve original masked tensors
        x1_out = x1_masked.clone()
        x2_out = x2_masked.clone()
        
        # Samples where BOTH modalities exist
        valid = (m1.view(-1) * m2.view(-1)).bool()        
        if valid.any():
            x1_out[valid] = self.cross_block(x1_masked[valid], x2_masked[valid])
            x2_out[valid] = self.cross_block(x2_masked[valid], x1_masked[valid])
        
        # Gating
        alpha1, alpha2, entropy = self.gating(x1_out, x2_out, m1.squeeze(), m2.squeeze())        
        Z_visible = alpha1 * x1_out + alpha2 * x2_out      
        return Z_visible, entropy, alpha1.squeeze(), alpha2.squeeze(), m1.squeeze(), m2.squeeze(), ids_restore, mask

    def decoder(self, Z_visible, ids_restore):
        if self.training:
            B = Z_visible.shape[0]
            N = self.decoder_pos_embed.shape[1]
            D = Z_visible.shape[2]
    
            mask_tokens = self.mask_token.expand(B, N - Z_visible.shape[1], D)
            Z_full = torch.cat([Z_visible, mask_tokens], dim=1)
            Z_full = torch.gather(Z_full, 1, ids_restore.unsqueeze(-1).repeat(1,1,D))
        else:
            Z_full = Z_visible
    
        Z_full = Z_full + self.decoder_pos_embed
        
        out1 = self.decoder_1(Z_full)
        out2 = self.decoder_2(Z_full)
        return out1, out2    

    def forward(self, img1, img2, m1=None, m2=None):  
        Z_visible, entropy, alpha1, alpha2, m1, m2, ids_restore, mask = self.encoder(img1, img2, m1, m2)
        out1, out2 = self.decoder(Z_visible, ids_restore)    
        return out1, out2, Z_visible, alpha1, alpha2, entropy, m1, m2, mask


# ------------------------
# Load Datasets
# ------------------------
from load_train_test_datasets import load_train_test_datasets
loader_train, loader_test = load_train_test_datasets()

# ------------------------
# Load Model
# ------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MultiModalMAE_MM(embed_dim=64, mask_ratio=0.5).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
n_epochs = 50

loss_vals = []
for epoch in range(n_epochs):
    total_loss = 0
    model.train()

    for img1, img2, _, _ in loader_train:
        img1 = img1.to(device)
        img2 = img2.to(device)

        pred1, pred2, _, _, _, entropy, m1, m2, mask = model(img1, img2)

        # create target patches
        def get_patches(img):
            patches = img.unfold(2,7,7).unfold(3,7,7)
            patches = patches.contiguous().view(img.size(0),16,-1)
            return patches

        target1 = get_patches(img1)
        target2 = get_patches(img2)        

        loss1 = (F.mse_loss(pred1, target1, reduction='none').mean(dim=[1,2]) * m1).sum() / m1.sum()
        loss2 = (F.mse_loss(pred2, target2, reduction='none').mean(dim=[1,2]) * m2).sum() / m2.sum()  
        
        lambda_entropy = 0.01  
        valid_entropy = (m1 * m2).mean()
        loss = loss1 + loss2 - lambda_entropy * entropy * valid_entropy          
        
        # loss1 = F.mse_loss(pred1, target1)
        # loss2 = F.mse_loss(pred2, target2)
        
        # lambda_entropy = 0.01  
        # loss = loss1 + loss2 - lambda_entropy * entropy

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()       

    print(f"Epoch {epoch + 1} Loss {total_loss/len(loader_train)}")
    loss_vals.append(total_loss)
    
plt.figure()
plt.plot(np.arange(1, n_epochs+1), loss_vals)
plt.xlabel('Epochs')
plt.ylabel('Total Loss')
plt.show()

torch.save(model.state_dict(), 'MAE_MultiModal_MM.pth')

model.load_state_dict(torch.load('MAE_MultiModal_MM.pth'))
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

def visualize_multimodal_reconstruction(model, loader, device, mode="both"):
    model.eval()

    img1, img2, _, _ = next(iter(loader))

    img1 = img1.to(device)
    img2 = img2.to(device)

    B = img1.size(0)

    if mode == "both":
        m1 = torch.ones(B, device=device)
        m2 = torch.ones(B, device=device)
    elif mode == "mnist_only":
        m1 = torch.ones(B, device=device)
        m2 = torch.zeros(B, device=device)
    elif mode == "fashion_only":
        m1 = torch.zeros(B, device=device)
        m2 = torch.ones(B, device=device)
    else:
        raise ValueError("mode must be 'both', 'mnist_only', or 'fashion_only'")

    with torch.no_grad():
        pred1, pred2, _, _, _, entropy, m1, m2, mask = model(img1, img2, m1, m2)

    recon1 = patches_to_image(pred1.cpu())
    recon2 = patches_to_image(pred2.cpu())

    img1 = img1.cpu()
    img2 = img2.cpu()

    num_show = 12
    fig, axes = plt.subplots(4, num_show, figsize=(8, 3))
    for i in range(num_show):
        axes[0,i].imshow(img1[i,0], cmap='gray')
        axes[0,i].axis('off')

        axes[1,i].imshow(recon1[i,0], cmap='gray')
        axes[1,i].axis('off')

        axes[2,i].imshow(img2[i,0], cmap='gray')
        axes[2,i].axis('off')

        axes[3,i].imshow(recon2[i,0], cmap='gray')
        axes[3,i].axis('off')
    plt.tight_layout()
    plt.show()

visualize_multimodal_reconstruction(model, loader_train, device, mode="both")
visualize_multimodal_reconstruction(model, loader_train, device, mode="mnist_only")
visualize_multimodal_reconstruction(model, loader_train, device, mode="fashion_only")
