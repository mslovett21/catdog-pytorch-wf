import glob, os
import random
import torch
from skimage import io, transform
from torch.utils.data import Dataset



class CatDogsDataset(Dataset):
    
    def __init__(self, filespath_class0, filepath_class1, transform = None):
        self.labels = {}
        self.filenames = []
        self.transform = transform
        
        class0_files = glob.glob(filespath_class0)
        class1_files = glob.glob(filepath_class1)
        
        self.__datasetup__(class0_files,0)
        self.__datasetup__(class1_files,1)
        
        random.shuffle(self.filenames)
    
    def __datasetup__(self,files, label):
        for filename in files:
            self.labels[filename] = label
            self.filenames.append(filename)


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        label = self.labels[filename]
        img = io.imread(filename)
        sample = {"image" : img,"label": label}
        
        if self.transform:
            sample = self.transform(sample)
        return sample


