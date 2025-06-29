from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

class CityScapes(Dataset):
    def __init__(self, data_path, split='train', transform=None, label_transform=None):
        super(CityScapes, self).__init__()

        self.image_path = os.path.join(data_path, "images", split)
        self.label_path = os.path.join(data_path, "gtFine", split)

        self.images_list = [
            os.path.relpath(os.path.join(root, file), start=self.image_path)
            for root, _, files in os.walk(self.image_path)
            for file in files if file.endswith(".png")
        ]
        self.labels_list = [
            os.path.relpath(os.path.join(root, file), start=self.label_path)
            for root, _, files in os.walk(self.label_path)
            for file in files if file.endswith("labelTrainIds.png")
        ]

        self.images_list.sort()
        self.labels_list.sort()

        self.transform = transform
        self.target_transform = label_transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_path, self.images_list[idx])
        label_path = os.path.join(self.label_path, self.labels_list[idx])

        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return {'x': image, 'y': label}