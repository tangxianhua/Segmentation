import os
import re
import numpy as np
import cv2
import scipy.io as sio
import skimage.io as io
import torch
from torch.utils.data.dataset import Dataset

class DataVnetConcate(Dataset):
    __im = []
    __imy = []
    __imz = []
    __mask = []
    dataset_size = 0

    def __init__(self, dataset_folder, train=True, transform=None):

        self.__file = []
        self.__im = []
        self.__imy = []
        self.__imz = []
        self.__mask = []
        self.transform = transform
        folder = dataset_folder
        if train:
            folder = os.path.join(r'/home/fafafa/pytorch3d/Data/', "mhdtrain")
        else:
            folder = os.path.join(r'/home/fafafa/pytorch3d/Data/', "mhdtest")

        for file in os.listdir(folder):
            if file.endswith(".mhd"):
                filename = os.path.splitext(file)[0]
                self.__file.append(filename)
                self.__im.append(os.path.join(folder, file))
                self.__imy.append(os.path.join(folder, file))
                self.__imz.append(os.path.join(folder, file))
        Mask_path = r"/home/fafafa/pytorch3d/Data/mhdmask"
        for mask_file in os.listdir(Mask_path):
            if mask_file.endswith(".mhd"):
                self.__mask.append(os.path.join(Mask_path, mask_file))

        self.dataset_size = len(self.__file)

    def __getitem__(self, index):

        img = io.imread(self.__im[index], plugin='simpleitk')
        imgy = io.imread(self.__imy[index], plugin='simpleitk')
        imgz = io.imread(self.__imz[index], plugin='simpleitk')
        mask = io.imread(self.__mask[index], plugin='simpleitk')

        img = np.expand_dims(img, axis=0)
        imgy = np.expand_dims(imgy, axis=0)
        imgz = np.expand_dims(imgz, axis=0)
        imgy = np.transpose(imgy, (0, 2, 1, 3))
        imgz = np.transpose(imgz, (0, 3, 2, 1))
        mask = np.expand_dims(mask, axis=0)

        img_x = torch.Tensor(img)
        img_y = torch.Tensor(imgy)
        img_z = torch.Tensor(imgz)
        mask_tr = torch.Tensor(mask)

        return img_x, img_y, img_z, mask_tr
    def __len__(self):
        return len(self.__im)
class DataVnet(Dataset):
    __file = []
    __im = []
    __mask = []

    dataset_size = 0

    def __init__(self, dataset_folder, train=True, transform=None):

        self.__file = []
        self.__im = []
        self.__mask = []
        self.transform = transform

        folder = dataset_folder
        # # Open and load text file including the whole training data
        if train:
            # folder = dataset_folder + "/Train/"
            folder = os.path.join(r"/home/fafafa/pytorch3d/Data/", "mhdtrain")
        else:
            folder = os.path.join(r"/home/fafafa/pytorch3d/Data/", "mhdtest")

        Maskfolder = r"/home/fafafa/pytorch3d/Data/mhdmask"
        for file in os.listdir(folder):
            if file.endswith(".mhd"):
                filename = os.path.splitext(file)[0]
                self.__file.append(filename)
                self.__im.append(os.path.join(folder, file))

        for maskfile in os.listdir(Maskfolder):
            if maskfile.endswith(".mhd"):
                self.__mask.append(os.path.join(Maskfolder, maskfile))
        self.dataset_size = len(self.__file)

    def __getitem__(self, index):

        img = io.imread(self.__im[index], plugin='simpleitk')
        mask = io.imread(self.__mask[index], plugin='simpleitk')

        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)

        img_tr = torch.Tensor(img)
        mask_tr = torch.Tensor(mask)

        return img_tr, mask_tr
        # return img.float(), mask.float()

    def __len__(self):

        return len(self.__im)



