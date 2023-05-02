import torch
import os
import numpy as np
import imageio.v3 as iio
import re

PARSE_REGEX = re.compile(r'image_(\d+).png')


class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = [os.path.join(main_dir, os.path.join(i, j))  for i in os.listdir(main_dir) for j in os.listdir(os.path.join(main_dir, i)) if "png" in j]
        self.all_msks = [(os.path.join(main_dir, os.path.join(i, "mask.npy")), int(PARSE_REGEX.match(j).group(1)))  for i in os.listdir(main_dir) for j in [ i for i in os.listdir(os.path.join(main_dir, i)) if "png" in i]]

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        image = iio.imread(self.all_imgs[idx])
        path, idx = self.all_msks[idx]
        mask = torch.from_numpy(np.load(path)[idx].astype(int))
        tensor_image = self.transform(image)
        return tensor_image, mask