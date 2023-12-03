
import random as rand
from glob import iglob
from typing import List

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from LightField import LightField


# from multipledispatch import dispatch

class DataSet:
    def __init__(self, params):
        self.num_views_hor: params.num_views_hor
        self.num_views_ver: params.num_views_ver
        self.resol_ver: params.resol_ver
        self.resol_hor: params.resol_hor
        self.bit_depth = params.bit_depth
        self.path = params.dataset_path
        self.limit_train = params.limit_train
        self.list_lfs = LazyList([], transforms = [ToTensor()])
        self.list_train = LazyList([], transforms = [ToTensor()])
        self.list_test = LazyList([], transforms = [ToTensor()])
        self.test_lf_names = ["Bikes", "Danger_de_Mort", "Ankylosaurus_&_Diplodocus_1", "Black_Fence", "Ceiling_Light", "Friends_1", "Houses_&_Lake", "Reeds",
                              "Rusty_Fence", "Slab_&_Lake", "Swans_2", "Vespa"]

        try:
            self.load_paths()
        except RuntimeError as e:
            print("Failed to load LFs: ", e)
            exit(11)

        if len(self.list_lfs) == 0:
            print("Failed to find LFs at path: ", self.path)
            exit(12)



    # TODO add new dataset variables at __str__
    def __str__(self):
        return ', '.join([self.path])

    def load_paths(self):
        count=0
        for lf_path in iglob(f"{self.path}/*/*"):
            self.list_lfs.append(LightField(lf_path))
            count+=1


    @classmethod
    def random_split(cls, list_lfs : list, train_percentage: float):
        train_size = int(len(list_lfs) * train_percentage)
        shuffled_lfs = rand.shuffle(list_lfs)
        list_train = shuffled_lfs[:train_size]
        list_validation = shuffled_lfs[train_size:]
        return (list_train, list_validation)

    # @dispatch()
    def split(self):
        for lf in self.list_lfs.inner_storage:
            if lf.name not in self.test_lf_names:
                if (len(self.list_train) < self.limit_train or self. limit_train == -1 ):
                    self.list_train.append(lf)
            else:
                self.list_test.append(lf)

        if(len(self.list_test) != len(self.test_lf_names)):
            print("Failed to find all test cases!!")
            exit(404)



class LazyList(Dataset):
    def __init__(self, inner_storage : List, transforms):
        self.inner_storage = inner_storage
        self.transforms = transforms
    def append(self, elem : LightField):
        self.inner_storage.append(elem)
    def __getitem__(self, i_index):
        X = self.inner_storage[i_index]
        X = X.load_lf()
        for transform in self.transforms:
            X = transform(X)
        return X
    def __len__(self):
        return len(self.inner_storage)

class LensletBlockedReferencer(Dataset):
    # possible TODO: make MI_size take a tuple
    def __init__(self, decoded, original, MI_size, predictor_size=32):
        super().__init__()
        self.decoded = decoded[0, :1, :, :]
        self.original = original[0, :1, :, :]
        self.predictor_size = predictor_size * MI_size
        self.inner_shape = decoded.shape
        assert(self.decoded.shape == self.original.shape)
        self.shape = tuple(dim // self.predictor_size - 1 for dim in self.inner_shape[-2:])
        assert(all(dim != 0 for dim in self.shape))
        self.len = self.shape[0] * self.shape[1]
    def __len__(self):
        return self.len
    def __getitem__(self, x):
        if x < -len(self) or x >= len(self):
            raise IndexError(x)
        elif x < 0:
            x += len(self)
        i, j = (x % self.shape[0], x // self.shape[0])
        section = self.decoded[:, i * self.predictor_size: (i + 2) * self.predictor_size, j * self.predictor_size: (j + 2) * self.predictor_size]
        """neighborhood = torch.ones(section.shape[0] + 1, *section.shape[1:], dtype=torch.float32)
        neighborhood[:-1, :, :, :, :] = section.to(neighborhood)
        neighborhood[:, :, :, self.predictor_size:, self.predictor_size:] = 0
        expected_block = self.original[:, :, :, i * self.predictor_size : (i+2) * self.predictor_size, j * self.predictor_size : (j+2) * self.predictor_size].to(neighborhood)"""

        neighborhood = torch.zeros(section.shape[0], *section.shape[1:], dtype=torch.float32)
        # print("neighborhood", neighborhood.shape)


        neighborhood[:, :, :] = section.to(neighborhood)

        avgtop = neighborhood[:, :self.predictor_size, :].mean()
        avgleft = neighborhood[:, :self.predictor_size, :self.predictor_size].mean()
        # neighborhood[:, self.predictor_size:, self.predictor_size:] = neighborhood[:, :self.predictor_size, :self.predictor_size].flip((-1, -2))
        neighborhood[:, self.predictor_size:, self.predictor_size:] = torch.full((self.predictor_size, self.predictor_size),
                                                                                 (avgleft + avgtop) / 2)



        print(neighborhood[:, self.predictor_size:, self.predictor_size:])
        expected_block = self.original[:, i * self.predictor_size: (i + 2) * self.predictor_size, j * self.predictor_size: (j + 2) * self.predictor_size].to(neighborhood)
        #print(neighborhood.shape)
        return neighborhood, expected_block
