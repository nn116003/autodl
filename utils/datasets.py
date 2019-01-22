import torchvision.datasets as datasets
import torch.utils.data as data

import pathlib

import pandas as pd

from PIL import Image

import os

class ImageFolderWithPathNoLabel(data.Dataset):
    def __init__(self, datadir, transform=None):
        self.pathes = list(pathlib.Path(datadir).iterdir())
        self.transform = transform

    def __getitem__(self, index):
        path = str(self.pathes[index])
        
        sample = Image.open(path).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, path

    def __len__(self):
        return len(self.pathes)
        

class ImageFolderWithPath(datasets.ImageFolder):
    def __getitem__(self, index):
        return super(ImageFolderWithPath, self).__getitem__(index), self.imgs[index][0]


class ImageWithScore(data.Dataset):
    def __init__(self, score_data, root='./', transform=None,
                 target_transform=None, with_path=False):
        self.score_data = pd.read_csv(score_data)
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.with_path = with_path
        
    def __getitem__(self, index):
        path = self.score_data.iloc[index, 0]
        score = self.score_data.iloc[index, 1]
        sample = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(score)

        if self.with_path:
            return sample, target, path
        else:
            return sample, target

    def __len__(self):
        return self.score_data.shape[0]

    def score_stats(self):
        scores = self.score_data.iloc[:,1]
        return scores.mean(), scores.std()
