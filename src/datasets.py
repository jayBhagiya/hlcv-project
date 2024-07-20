import cv2
import matplotlib.pyplot as plt
import os
import shutil

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms 

class CustomDataset(Dataset):
    """Custom dataset to read from dataframe and return image and its respective labels

    Args:
        Dataset (torch.data.Dataset): Subclass the pytorch Dataset class
    """
    def __init__(self, real_folder, synthetic_folder, transform=None):
        self.real_folder = real_folder
        self.synthetic_folder = synthetic_folder
        assert sorted(os.listdir(real_folder)) == sorted(os.listdir(synthetic_folder))
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.real_folder))

    def __getitem__(self, index):
        image_name = sorted(os.listdir(self.real_folder))[index]
        real_path = os.path.join(self.real_folder, image_name)
        synthetic_path = os.path.join(self.synthetic_folder, image_name)

        real_image = cv2.imread(real_path)
        synthetic_image = cv2.imread(synthetic_path)
        real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)
        synthetic_image = cv2.cvtColor(synthetic_image, cv2.COLOR_BGR2RGB)
        if self.transform:
            real_image = self.transform(real_image)
            synthetic_image = self.transform(synthetic_image)
        return synthetic_image, real_image
