from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import torch.utils.data as data
from PIL import Image
import PIL
import os
import os.path
import pickle
import random
import numpy as np
import pandas as pd
import glob
from miscc.config import cfg

class FolderStoryDataset(data.Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.folders = [ story for story in glob.glob(os.path.join(data_dir, '*')) ]
        self.folders.sort()
        self.transform = transform

    def __len__(self):
        return len(self.folders)
    
    def __getitem__(self,index):
        folder = self.folders[index]
        stories = []
        for idx in range(5):
            img_path = os.path.join(folder, str(idx)+'.png')
            img = PIL.Image.open(img_path)
            img = img.convert('RGB')
            stories.append(np.array(img))
        stories = np.array(stories)
        return self.transform(stories)

class FolderImageDataset(data.Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.folders = [ story for story in glob.glob(os.path.join(data_dir, '*/*.png')) ]
        self.folders.sort()
        self.transform = transform

    def __len__(self):
        return len(self.folders)
    
    def __getitem__(self,index):
        folder = self.folders[index]
        img = PIL.Image.open(folder)
        img = img.convert('RGB')
        img = np.array(img)
        return self.transform(img)
