import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    #from sklearn.cross_validation import StratifiedShuffleSplit
    # cross_validation -> now called: model_selection
    # https://stackoverflow.com/questions/30667525/importerror-no-module-named-sklearn-cross-validation
    from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

    
class CelebADataset(Dataset):
    
    def __init__(self, csv_path, image_path, image_shape, target):
        csv_file = pd.read_csv(csv_path)
        out = np.zeros([len(csv_file), 2])
        for row, col in enumerate(csv_file[target]):
            out[int(row), int(col)] = 1
        self.target = out
        self.attributes = csv_file.drop(target,axis=1)
        self.image_path = image_path
        self.image_shape = image_shape
        
    def __len__(self):
        return len(self.target)    
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_path,
                                self.attributes.iloc[idx, 0])
        image = imread(img_name)
        image = resize(image, output_shape=self.image_shape, mode='reflect', anti_aliasing=True)
        target = self.target[idx]
        image = image.transpose((2, 0, 1))
        attributes = self.attributes.iloc[idx,1:-1]
        attributes = np.array(attributes).astype('float')
        sample = {'image': torch.from_numpy(image.astype('float32')), 'target': torch.from_numpy(target), 'attributes':torch.from_numpy(attributes)}

        return sample
        