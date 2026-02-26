import torch
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import random


class PairedDataset(Dataset):
    def __init__(self, dataset1, dataset2_by_label):
        self.dataset1 = dataset1
        self.dataset2_by_label = dataset2_by_label

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, idx):
        img1, label1 = self.dataset1[idx]

        # Match same label (0->0, 1->1, ..., 9->9)
        img2 = random.choice(self.dataset2_by_label[label1])
        label2 = label1
        return img1, img2, label1, label2


def load_train_datasets():
    # Training Datasets
    dataset1 = torchvision.datasets.MNIST(
        './data', train=True, download=True, transform=transforms.ToTensor()
    )
    dataset2 = torchvision.datasets.FashionMNIST(
        './data', train=True, download=True, transform=transforms.ToTensor()
    )
    
    # ---- Select 10,000 random indices ----
    num_train_samples = 10000
    indices = torch.randperm(len(dataset1))[:num_train_samples]
    
    dataset1_subset = Subset(dataset1, indices)
    
    dataset2_by_label = defaultdict(list)
    for i in indices:
        img, label = dataset2[i]
        dataset2_by_label[label].append(img)
    
    paired_dataset = PairedDataset(dataset1_subset, dataset2_by_label)
    loader_train = DataLoader(paired_dataset, batch_size=128, shuffle=True)
    return loader_train


def load_test_datasets():
    # Training Datasets
    dataset1 = torchvision.datasets.MNIST(
        './data', train=False, download=True, transform=transforms.ToTensor()
    )
    dataset2 = torchvision.datasets.FashionMNIST(
        './data', train=False, download=True, transform=transforms.ToTensor()
    )
    
    # ---- Select 10,000 random indices ----
    num_train_samples = 1000
    indices = torch.randperm(len(dataset1))[:num_train_samples]
    
    dataset1_subset = Subset(dataset1, indices)
    
    dataset2_by_label = defaultdict(list)
    for i in indices:
        img, label = dataset2[i]
        dataset2_by_label[label].append(img)
    
    paired_dataset = PairedDataset(dataset1_subset, dataset2_by_label)
    loader_test = DataLoader(paired_dataset, batch_size=128, shuffle=True)
    return loader_test
    
def load_train_test_datasets():
    loader_train = load_train_datasets()
    loader_test = load_test_datasets()    
    # loader train: img1, img2, label1, label2
    # loader test: img1, img2, label1, label2    
    return loader_train, loader_test



