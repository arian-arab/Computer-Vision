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
            
class MultiModalMAE_MM(nn.Module):
    def __init__(self, embed_dim=64, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.patch_embed = PatchEmbed(embed_dim=embed_dim)
        
        num_patches = self.patch_embed.num_patches

        self.encoder_pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))

        self.block1 = CrossAttentionBlock(embed_dim)
        self.block2 = CrossAttentionBlock(embed_dim)

        self.gating = GatingModule_MM(embed_dim)
        
        self.decoder_pos_embed_1 = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        self.decoder_pos_embed_2 = nn.Parameter(torch.randn(1, num_patches, embed_dim))

        self.decoder_1 = Decoder(embed_dim)
        self.decoder_2 = Decoder(embed_dim)        
    
    def random_masking(self, x):
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1,1,D))
        return x_masked, ids_restore, ids_keep
    
    def encoder(self, img1, img2, m1=None, m2=None):
        B = img1.size(0)
        device = img1.device    
        
        if self.training:
            probs = torch.rand(B, 1, device=device)
    
            m1 = torch.ones(B, 1, device=device)
            m2 = torch.ones(B, 1, device=device)
    
            m1[probs < 0.33] = 1
            m2[probs < 0.33] = 0
    
            m1[(probs >= 0.33) & (probs < 0.66)] = 0
            m2[(probs >= 0.33) & (probs < 0.66)] = 1
        else:
            m1 = m1.view(B,1).to(device)
            m2 = m2.view(B,1).to(device)              
        
        # Cross interaction on sparse tokens  
        x1 = self.patch_embed(img1) + self.encoder_pos_embed
        x2 = self.patch_embed(img2) + self.encoder_pos_embed
        
        if self.training:
            # Shared masking
            x1_masked, ids_restore, ids_keep = self.random_masking(x1)        
            # Apply same indices to x2
            x2_masked = torch.gather(x2, dim=1, index=ids_keep.unsqueeze(-1).repeat(1,1,x2.shape[2]))  
        else:
            x1_masked = x1
            x2_masked = x2
            ids_restore = None
            ids_keep = None
            
        # Zero tokens if modality absent
        x1_masked = x1_masked * m1.unsqueeze(-1)
        x2_masked = x2_masked * m2.unsqueeze(-1)
        
        # Cross interaction on sparse tokens        
        x1_out = self.block1(x1_masked, x2_masked) if m1.sum() > 0 else x1_masked
        x2_out = self.block2(x2_masked, x1_masked) if m2.sum() > 0 else x2_masked
        
        # Gating
        alpha1, alpha2, entropy = self.gating(x1_out, x2_out, m1.squeeze(), m2.squeeze())        
        Z_visible = alpha1 * x1_out + alpha2 * x2_out      
        return Z_visible, entropy, alpha1.squeeze(), alpha2.squeeze(), m1.squeeze(), m2.squeeze(), ids_restore, ids_keep

    def decoder(self, Z_visible, ids_restore):
        if self.training:
            B = Z_visible.shape[0]
            N = self.decoder_pos_embed_1.shape[1]
            D = Z_visible.shape[2]
    
            mask_tokens = self.mask_token.expand(B, N - Z_visible.shape[1], D)
            Z_full = torch.cat([Z_visible, mask_tokens], dim=1)
            Z_full = torch.gather(Z_full, 1, ids_restore.unsqueeze(-1).repeat(1,1,D))
        else:
            Z_full = Z_visible
    
        Z_full_1 = Z_full + self.decoder_pos_embed_1
        Z_full_2 = Z_full + self.decoder_pos_embed_2
        
        out1 = self.decoder_1(Z_full_1)
        out2 = self.decoder_2(Z_full_2)
        return out1, out2    

    def forward(self, img1, img2, m1=None, m2=None):  
        Z_visible, entropy, alpha1, alpha2, m1, m2, ids_restore, ids_keep = self.encoder(img1, img2, m1, m2)
        out1, out2 = self.decoder(Z_visible, ids_restore)    
        return out1, out2, Z_visible, alpha1, alpha2, entropy, m1, m2


# ------------------------
# Load Datasets
# ------------------------
from load_train_test_datasets import load_train_test_datasets
loader_train, loader_test = load_train_test_datasets()

# ------------------------
# Load Model
# ------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MultiModalMAE_MM(embed_dim=256, mask_ratio=0.75).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
n_epochs = 100

loss_vals = []
for epoch in range(n_epochs):
    total_loss = 0
    model.train()

    for img1, img2, _, _ in loader_train:
        img1 = img1.to(device)
        img2 = img2.to(device)

        pred1, pred2, _, _, _, entropy, m1, m2 = model(img1, img2)

        # create target patches
        def get_patches(img):
            patches = img.unfold(2,7,7).unfold(3,7,7)
            patches = patches.contiguous().view(img.size(0),16,-1)
            return patches

        target1 = get_patches(img1)
        target2 = get_patches(img2)
        
        # loss1 = F.mse_loss(pred1, target1)
        # loss2 = F.mse_loss(pred2, target2)
        
        # lambda_entropy = 0.01  
        # loss = loss1 + loss2 - lambda_entropy * entropy

        loss1 = (F.mse_loss(pred1, target1, reduction='none').mean(dim=[1,2]) * m1).sum() / m1.sum()
        loss2 = (F.mse_loss(pred2, target2, reduction='none').mean(dim=[1,2]) * m2).sum() / m2.sum()  
        
        lambda_entropy = 0.01  
        valid_entropy = (m1 * m2).mean()
        loss = loss1 + loss2 - lambda_entropy * entropy * valid_entropy        

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
def visualize_images(loader):
    img1, img2, _, _ = next(iter(loader))
    img1 = img1.cpu()
    img2 = img2.cpu()

    num_show = 12
    fig, axes = plt.subplots(2, num_show, figsize=(8, 3))
    for i in range(num_show):
        axes[0,i].imshow(img1[i,0], cmap='gray')
        axes[0,i].axis('off')

        axes[1,i].imshow(img2[i,0], cmap='gray')        
        axes[1,i].axis('off')
        
    plt.tight_layout()    
    plt.show()

visualize_images(loader_test)


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
        pred1, pred2, _, _, _, _, _, _ = model(img1, img2, m1, m2)

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


def collect_alphas(model, loader, device, mode="both"):
    model.eval()

    alpha1_list = []
    alpha2_list = []

    with torch.no_grad():
        for img1, img2, _, _ in loader:

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
                raise ValueError("Invalid mode")

            _, _, _, a1, a2, _, _, _ = model(img1, img2, m1, m2)

            alpha1_list.append(a1.cpu())
            alpha2_list.append(a2.cpu())

    alpha1_all = torch.cat(alpha1_list)
    alpha2_all = torch.cat(alpha2_list)

    return alpha1_all, alpha2_all

alpha1_all, alpha2_all = collect_alphas(model, loader_train, device, mode="both")
plt.hist(alpha1_all.numpy(), bins=50, alpha=0.6, label="alpha1 (MNIST)")
plt.hist(alpha2_all.numpy(), bins=50, alpha=0.6, label="alpha2 (Fashion)")
plt.legend()
plt.title("Gating Weights Distribution - MNIST Only")
plt.show()


# ------------------------
# Extract Embeddings
# ------------------------
model.eval()

all_embeddings = []
all_labels = []
with torch.no_grad():
    for img1, img2, labels, _ in loader_test:

        img1 = img1.to(device)
        img2 = img2.to(device)

        B = img1.size(0)

        m1 = torch.ones(B, device=device)
        m2 = torch.ones(B, device=device) 
        
        _, _, Z, _, _, _, _, _ = model(img1, img2, m1, m2)
        features = Z.mean(dim=1)

        all_embeddings.append(features.cpu())
        all_labels.append(labels)

all_embeddings = torch.cat(all_embeddings, dim=0)
all_labels = torch.cat(all_labels, dim=0)

print("Train shape:", all_embeddings.shape)

X = all_embeddings.detach().cpu().numpy()
y_true = all_labels.detach().cpu().numpy()

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10, random_state=42, n_init=20)
y_kmeans = kmeans.fit_predict(X)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='tab10', s=30)
plt.title("Ground Truth Labels (PCA projection)")
plt.colorbar()
plt.show()

from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

# Compute confusion matrix
cm = confusion_matrix(y_true, y_kmeans)

# Hungarian algorithm to match clusters to labels
row_ind, col_ind = linear_sum_assignment(-cm)

# Create mapping
mapping = {col: row for row, col in zip(row_ind, col_ind)}

# Remap cluster labels
y_kmeans_aligned = np.vectorize(mapping.get)(y_kmeans)

cm_aligned = confusion_matrix(y_true, y_kmeans_aligned)
print("Confusion Matrix After Alignment:")
print(cm_aligned)

accuracy = np.trace(cm_aligned) / np.sum(cm_aligned)
print(f"Clustering Accuracy: {accuracy:.4f}")