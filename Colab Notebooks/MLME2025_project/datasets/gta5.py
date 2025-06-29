from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import torch
import random
from abc import ABCMeta
from dataclasses import dataclass
from typing import Tuple
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
import torch.nn.functional as F


class GTA5(Dataset):
    def __init__(self, data_path, transform=None, label_transform=None, aug_transform=None, FDA_transform=None):
        super(GTA5, self).__init__()

        self.image_path = data_path + "images/"
        self.label_path = data_path + "labels/"

        self.images_list = [x for x in os.listdir(self.image_path) if x.endswith(".png")]
        self.labels_list = [x for x in os.listdir(self.label_path) if x.endswith(".png")]

        self.images_list.sort()
        self.labels_list.sort()

        # transforms for images and labels
        self.transform = transform
        self.label_transform = label_transform
        self.aug_transform = aug_transform
        self.FDA_transform = FDA_transform

        # label metadata (class IDs and colors)
        self.labels = GTA5Labels_TaskCV2017()
        self.valid_ids = self.labels.support_id_list
        self.color_to_trainid = {label.color: label.ID for label in self.labels.list_}

    def __len__(self):
      return len(self.images_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_path, self.images_list[idx])
        label_path = os.path.join(self.label_path, self.labels_list[idx])

        image = Image.open(img_path).convert('RGB')
        label_rgb = Image.open(label_path).convert('RGB')

        # Convert RGB label image to class ID label
        label = self.rgb_to_trainid(label_rgb)
        
        # Apply FDA transform if provided
        if self.FDA_transform:
            image = self.FDA_transform(image)

        # Apply joint image-label augmentation if provided
        if self.aug_transform:
            image, label = self.aug_transform(image, label)

        # Apply image-only transform
        if self.transform:
            image = self.transform(image)
        
        # Apply label-only transform
        if self.label_transform:
            label = self.label_transform(label)

        return {'x': image, 'y': label}

    def rgb_to_trainid(self, label_img):
        """
        Converts an RGB segmentation map to a label map with train IDs.
        Unknown classes are assigned the value 255 (ignore index).
        """
        label_np = np.array(label_img)
        h, w, _ = label_np.shape
        label_id = np.full((h, w), 255, dtype=np.uint8)

        for color, trainid in self.color_to_trainid.items():
            mask = np.all(label_np == color, axis=-1)
            label_id[mask] = trainid

        return Image.fromarray(label_id, mode='L')




class BaseGTALabels(metaclass=ABCMeta):
    pass


@dataclass
class GTA5Label:
    ID: int
    color: Tuple[int, int, int]


class GTA5Labels_TaskCV2017(BaseGTALabels):
    road = GTA5Label(ID=0, color=(128, 64, 128))
    sidewalk = GTA5Label(ID=1, color=(244, 35, 232))
    building = GTA5Label(ID=2, color=(70, 70, 70))
    wall = GTA5Label(ID=3, color=(102, 102, 156))
    fence = GTA5Label(ID=4, color=(190, 153, 153))
    pole = GTA5Label(ID=5, color=(153, 153, 153))
    light = GTA5Label(ID=6, color=(250, 170, 30))
    sign = GTA5Label(ID=7, color=(220, 220, 0))
    vegetation = GTA5Label(ID=8, color=(107, 142, 35))
    terrain = GTA5Label(ID=9, color=(152, 251, 152))
    sky = GTA5Label(ID=10, color=(70, 130, 180))
    person = GTA5Label(ID=11, color=(220, 20, 60))
    rider = GTA5Label(ID=12, color=(255, 0, 0))
    car = GTA5Label(ID=13, color=(0, 0, 142))
    truck = GTA5Label(ID=14, color=(0, 0, 70))
    bus = GTA5Label(ID=15, color=(0, 60, 100))
    train = GTA5Label(ID=16, color=(0, 80, 100))
    motocycle = GTA5Label(ID=17, color=(0, 0, 230))
    bicycle = GTA5Label(ID=18, color=(119, 11, 32))

    list_ = [
        road,
        sidewalk,
        building,
        wall,
        fence,
        pole,
        light,
        sign,
        vegetation,
        terrain,
        sky,
        person,
        rider,
        car,
        truck,
        bus,
        train,
        motocycle,
        bicycle,
    ]

    @property
    def support_id_list(self):
        ret = [label.ID for label in self.list_]
        return ret