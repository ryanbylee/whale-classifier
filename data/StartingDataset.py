import torch
import os
import pandas as pd
from torchvision.io import read_image
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from PIL import Image



class StartingDataset(torch.utils.data.Dataset):
    """
    Starting dataset contains 100000 3x224x224 black images (all zeros).
    Below is modified to take custom dataset.
    """

    
    def __init__(self, img_directory, dataMode = "train", df = None, transform = transforms.ToTensor(), labels = None):
        self.img_directory = img_directory
        self.dataMode = dataMode
        self.labels = labels
        if self.dataMode == "train":
            self.df = df.values
        self.image_list = [images for images in os.listdir(img_directory)]
        self.transform = transform        

    def __getitem__(self, index):
        if self.dataMode == "train":
            image_name = os.path.join(self.img_directory, self.df[index][0])
            label = self.labels[index]
        elif self.dataMode == "test":
            image_name = os.path.join(self.img_directory, self.image_list[index])
            label = np.zeros((5005, ))
        
        image = Image.open(image_name).convert("RGB")
        image = self.transform(image)

        if self.dataMode == "train":
            return image, label
        elif self.dataMode == "test":
            return image, label, self.image_list[index]


    def __len__(self):
        return len(self.image_list)
    
