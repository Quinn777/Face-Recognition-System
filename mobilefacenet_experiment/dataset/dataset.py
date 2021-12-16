import os
import numpy as np
from PIL import Image

from torch.utils import data
from torchvision import transforms as T


class Dataset(data.Dataset):

    def __init__(self, root, dataset_list, phase, input_shape=(112, 112), img_mode='RGB'):
        self.phase = phase
        self.img_mode = img_mode
        self.input_shape = input_shape


        with open(dataset_list, 'r') as fd:
            self.imgs_name = fd.readlines()

        # if phase == 'train':
        #     self.imgs_name = self.imgs_name[:2000]

        self.imgs = [os.path.join(root, img[:-1]) for img in self.imgs_name] # img[:-1] for '\n' is the end of the line

        normalize = None
        if self.img_mode == 'RGB':
            normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])
        elif self.img_mode == 'L':
            normalize = T.Normalize(mean=[0.5], std=[0.5])

        if self.phase == 'train':
            self.transforms = T.Compose([
                T.Resize(self.input_shape),
                # T.RandomCrop(self.input_shape),
                T.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.Resize(self.input_shape),
                # T.CenterCrop(self.input_shape),
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        sample = self.imgs[index]
        if self.phase == 'train':
            splits = sample.split(' ')
            img_path = splits[0]
            data = Image.open(img_path).convert(self.img_mode)
            data = self.transforms(data)
            label = np.int32(splits[1])
            return data.float(), label
        else:
            data = Image.open(sample).convert(self.img_mode)
            data = self.transforms(data)
            name = self.imgs_name[index][:-1]
            return data.float(), name

    def __len__(self):
        return len(self.imgs)
