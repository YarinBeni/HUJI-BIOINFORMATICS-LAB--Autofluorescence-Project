import os
import pandas as pd
import torch
from torch.utils.data import Dataset
# from skimage import io
import cv2 as cv
import torchvision.transforms as transforms



# csvfile formt imagename, imagelabel
class WormsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # in csv row index column 0 meaing name of image in row index in csv
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = cv.imread(img_path, 0)  # 0 is grey flag
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
        if self.transform:
            image = self.transform
        return image, y_label
