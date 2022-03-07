import glob
import cv2
import io
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader


class LOLDataTrainSet(Dataset):
    """
    rootpath:LOLdataset
    """
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.low_dir = glob.glob(root_dir+"\\our485\\low\\*.png")
        self.high_dir = glob.glob(root_dir+"\\our485\\high\\*.png")
        self.transform = transforms.Compose([transforms.ToTensor()])
    def __len__(self):
        return len(self.low_dir)
    def __getitem__(self, item):
        low = Image.open(self.low_dir[item])
        high = Image.open(self.high_dir[item])
        low = self.transform(low)
        high = self.transform(high)
        return low,high

if __name__ == '__main__':
    ds = LOLDataTrainSet(root_dir=".\LOLdataset")
    train_loader  =DataLoader(ds,batch_size = 1,shuffle=False,num_workers=1)
    for i, sample in enumerate(train_loader):
        if i == 100:
            low = sample[0]
            low = low.squeeze()
            low = low.permute((1,2,0)).numpy()
            plt.imshow(low)
            plt.show()
            break
