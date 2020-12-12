# -*- coding:utf8 -*-
# @TIME     : 2020/12/4 15:48
# @Author   : SuHao
# @File     : mvtec.py

'''
reference: https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
'''


import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import numpy as np

# URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']


class MVTecDataset(Dataset):
    def __init__(self, dataset_path='D:\硕士\mvtec', class_name='bottle', is_train=True, resize=128):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        # self.cropsize = cropsize

        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()



    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        # set transforms
        angle = np.random.randint(0, 360)
        self.transform_x = T.Compose([T.Resize(self.resize, Image.ANTIALIAS),
                                      T.ToTensor(),])
                                      # T.Normalize(mean=(0.5, 0.5, 0.5),
                                      #             std=(0.5, 0.5, 0.5))])
        self.transform_mask = T.Compose([T.Resize(self.resize, Image.NEAREST),
                                         T.ToTensor(),])
        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        if y == 0:
            mask = torch.zeros([1, self.resize, self.resize])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)


    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if (f.endswith('.png') or f.endswith('.jpg'))])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)
