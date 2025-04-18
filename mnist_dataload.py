import os
import gzip
import numpy as np
import torch
from torch.utils.data import Dataset

def load_images(path):
    with gzip.open(path, 'rb') as f:
        _ = int.from_bytes(f.read(4), 'big')  # magic number
        num = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, 1, rows, cols).astype(np.float32)

def load_labels(path):
    with gzip.open(path, 'rb') as f:
        _ = int.from_bytes(f.read(4), 'big')  # magic number
        num = int.from_bytes(f.read(4), 'big')
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.astype(np.int64)

# 2. Dataset class with ToTensor + Normalize
class MNISTDataset(Dataset):
    def __init__(self, images, labels, normalize=True):
        if normalize:
            images = (images / 255.0 - 0.1307) / 0.3081
        self.images = torch.from_numpy(images)  # [N, 1, 28, 28], float32
        self.labels = torch.from_numpy(labels)  # [N], int64

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# 3. Create the datasets (equivalent to dataset1, dataset2)
def get_datasets(data_dir='./data'):
    # download_mnist(data_dir)
    train_images = load_images(os.path.join(data_dir, 'MNIST/raw', 'train-images-idx3-ubyte.gz'))
    train_labels = load_labels(os.path.join(data_dir, 'MNIST/raw', 'train-labels-idx1-ubyte.gz'))
    test_images = load_images(os.path.join(data_dir, 'MNIST/raw', 't10k-images-idx3-ubyte.gz'))
    test_labels = load_labels(os.path.join(data_dir, 'MNIST/raw', 't10k-labels-idx1-ubyte.gz'))

    dataset1 = MNISTDataset(train_images, train_labels)
    dataset2 = MNISTDataset(test_images, test_labels)
    return dataset1, dataset2