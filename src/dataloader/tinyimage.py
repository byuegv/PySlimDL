import os
import torch
from PIL import Image
import numpy as np
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from skimage import io,transform
from torchvision.datasets import VisionDataset

# Ignores warnings
import warnings
warnings.filterwarnings("ignore")

class TinyImage(VisionDataset):
    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, class_num=100, train=True, transform=None, target_transform=None,
        download=False):
        super(CIFAR10, self).__init__(root,transform=transform,target_transform=target_transform)
        self.train = train
        self.class_num=class_num
        self.training_file = "tinyimage-{}-training.txt".format(class_num)
        self.test_file = "tinyimage-{}-test.txt".format(class_num)

        # data and target
        self.data = None
        self.targets = None

        if download:
            self.download()
        if self.train:
            data_file = self.training_file
            self.data, self.targets = self.load_agg_train_data(os.path.join(self.root,data_file))
        else:
            data_file = self.test_file
            self.data, self.targets = self.load_test_data(os.path.join(self.root,data_file))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    """Args:
        data_file: data file where aggregated data points stored

        file structure: aggregated_points_1 | original_points_1 | original_points_2 | ...
                        aggregated_points_2 | original_points_1 | original_points_2 | ...
        for each points: label; feature_1, feature,2, ...
    """
    def load_agg_train_data(self,data_file):
        _data = []
        _targets = []

        with open(data_file) as infile:
            for line in infile:
                nline = line.strip().split('|')
                sample_data = []
                sample_target = []
                #print("agg+orig: {}".format(len(nline)))
                for i in range(len(nline)):
                    sample = nline[i].strip().split(';')
                    target = int(float(sample[0])) - 1
                    features = list(map(lambda v: float(v),sample[1].split(',')))
                    sample_data.append(features)
                    sample_target.append(target)
                _data.append(sample_data)
                _targets.append(sample_target)
        return torch.tensor(_data), torch.tensor(_targets)

    """Args:
        data_file: data file where test points stored

        file structure: original_points_1
                        original_points_2
                        ...
        for each points: label; feature_1, feature,2, ...
    """
    def load_test_data(self,data_file):
        _data = []
        _targets = []
        with open(data_file) as infile:
            for line in infile:
                nline = line.strip().split(';')
                target = int(float(nline[0])) - 1
                features = list(map(lambda v: float(v),nline[1].split(',')))
                _targets.append(target)
                _data.append(features)
        return torch.tensor(_data), torch.tensor(_targets)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    def download(self):
        raise NotImplementedError


    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


class CriticalTinyImage(Dataset):
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, datax, targetsx, train=True, transform=None,target_transform=None):
        super(CriticalTinyImage, self).__init__()
        self.train = train

        self.transform = transform
        self.target_transform = transform

        # data and target
        self.data = datax
        self.targets = targetsx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
