
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
        self.list_lfs = LazyList([], transforms = [ToTensor()], bit_depth=self.bit_depth)
        self.list_train = LazyList([], transforms = [ToTensor()], bit_depth=self.bit_depth)
        self.list_test = LazyList([], transforms = [ToTensor()], bit_depth=self.bit_depth)
        self.test_lf_names = ["Bikes", "Danger_de_Mort", "Ankylosaurus_&_Diplodocus_1", "Black_Fence", "Ceiling_Light", "Friends_1", "Houses_&_Lake", "Reeds",
                              "Rusty_Fence", "Slab_&_Lake", "Swans_2", "Vespa"]
        #
        if params.limit_val != -1:
            self.test_lf_names = self.test_lf_names[:params.limit_val]

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
    def __init__(self, inner_storage : List, transforms, bit_depth = 8):
        self.inner_storage = inner_storage
        self.transforms = transforms
        self.bit_depth = bit_depth

    def append(self, elem : LightField):
        self.inner_storage.append(elem)
    def __getitem__(self, i_index):
        temp_lf = self.inner_storage[i_index]
        lf_array = temp_lf.load_lf()
        for transform in self.transforms:
            lf_array = transform(lf_array)
        return lf_array
    def __len__(self):
        return len(self.inner_storage)

class LensletBlockedReferencer(Dataset):
    # possible TODO: make MI_size take a tuple
    def __init__(self, original, MI_size, predictor_size=32, context_size=64, 
                 loss_mode="predOnly"):
        super().__init__()
        self.count=0
        self.decoded = original[0, :1, :, :]
        self.original = original[0, :1, :, :]
        self.predictor_size = predictor_size * MI_size
        self.context_size= context_size * MI_size
        self.inner_shape = original.shape
        self.loss_mode = loss_mode
        assert(self.decoded.shape == self.original.shape)
        self.shape = tuple(dim // self.context_size - 1 for dim in self.inner_shape[-2:])
        # print("inner", self.shape)
        assert(all(dim != 0 for dim in self.shape))
        self.len = self.shape[0] * self.shape[1]
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, batch_size):
        if batch_size < -len(self) or batch_size >= len(self):
            raise IndexError(batch_size)
        elif batch_size < 0:
            batch_size += len(self)
        i, j = (batch_size % self.shape[0], batch_size // self.shape[0])
        # print("i, j = ", i, j)
        # print("batch_size = ", batch_size)

        stepI = i * self.predictor_size
        stepJ = j * self.predictor_size
        section = self.decoded[:, stepI:stepI+self.context_size, stepJ:stepJ + self.context_size]

        # print("section ", section.shape)
        """neighborhood = torch.ones(section.shape[0] + 1, *section.shape[1:], dtype=torch.float32)
        neighborhood[:-1, :, :, :, :] = section.to(neighborhood)
        neighborhood[:, :, :, self.predictor_size:, self.predictor_size:] = 0
        expected_block = self.original[:, :, :, i * self.predictor_size : (i+2) * self.predictor_size, j * self.predictor_size : (j+2) * self.predictor_size].to(neighborhood)"""

        neighborhood = torch.zeros(section.shape[0], *section.shape[1:], dtype=torch.float32)
        


        neighborhood[:, :, :] = section.to(neighborhood)
        if self.loss_mode == "predOnly":
            expected_block = neighborhood[:, -self.predictor_size:, -self.predictor_size:].clone() #.to(neighborhood)
        elif self.loss_mode == "fullContext":
            expected_block = neighborhood[:, :, :].clone()
            #print(expected_block.shape)
        else: 
            print("ERROR Loss Mode Not Found", self.loss_mode)
            
        
        #if self.context_mode == 'avg':
         #   avgtop = neighborhood[:, :self.predictor_size, :].mean()
          #  avgleft = neighborhood[:, -self.predictor_size:, :self.predictor_size].mean()
           # neighborhood[:, -self.predictor_size:, -self.predictor_size:] = (avgleft+avgtop)/2
        #elif self.context_mode == 'black':
        neighborhood[:, -self.predictor_size:, -self.predictor_size:] = torch.zeros((self.predictor_size, self.predictor_size))
        #else: print("ERROR CONTEXT MODE NOT FOUND")


        #print(expected_block.shape)

       
        return neighborhood, expected_block
