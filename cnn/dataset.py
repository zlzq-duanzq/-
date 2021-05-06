#from torchvision import datasets, transforms
from skimage.color import rgb2lab, rgb2gray
#from skimage import io
import torch.utils.data as data
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
from util import *
import matplotlib.pyplot as plt

class ImageFolder(data.Dataset):
    def __init__(self, flag, dataDir='./images/', data_range=(0, 8), n_class=64, 
               onehot=False):
        self.onehot = onehot
        assert(flag in ['train', 'val', 'test'])
        print("load " + flag + " dataset start")
        print("    from: %s" % dataDir)
        print("    range: [%d, %d)" % (data_range[0], data_range[1]))
        self.dataset = []
        for i in range(data_range[0], data_range[1]):
            #img = Image.open(os.path.join(dataDir, flag, f'{i}.jpg'))
            #img = np.asarray(img)
            img = plt.imread(os.path.join(dataDir, flag, f'{i}.jpg'))
            img_gray = rgb2gray(img)
            img_gray = torch.from_numpy(img_gray).unsqueeze(0).float()
            img_lab = rgb2lab(img)
            img_lab = img_lab + 128
            img_ab = img_lab[:,:,1:3].transpose((2,0,1))
            idx = ab2idx(img_ab)
            label = np.zeros((n_class, img_ab.shape[1], img_ab.shape[2])).astype("i")
            for j in range(n_class):
                label[j, :] = idx == j
            
            self.dataset.append((img_gray, label))
        print("load dataset done")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        label = torch.FloatTensor(label)
        if not self.onehot:
            label = torch.argmax(label, dim=0)
        else:
            label = label.long()
        return torch.FloatTensor(img), torch.LongTensor(label)