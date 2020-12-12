# -*- coding:utf8 -*-
# @TIME     : 2020/12/11 17:43
# @Author   : SuHao
# @File     : skip_GANomaly_test.py

import os
import tqdm
import torch
import numpy as np
import cv2
import argparse
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from dataload.dataload import load_dataset
from models.DCGAN_skip_GANomaly import NetG, NetD
from test.evaluate import Evaluate, draw_heatmap



parser = argparse.ArgumentParser()
parser.add_argument("--experiment", default=r"../experiments/skip_GAnomaly_test", help="path to save experiments results")
parser.add_argument("--dataset", default="grid", help="mnist")
parser.add_argument('--dataroot', default=r"../../../mvtec", help='path to dataset')
parser.add_argument("--n_epoches", type=int, default=1, help="number of epoches of training")
parser.add_argument("--batchSize", type=int, default=1, help="size of the batches")
parser.add_argument("--size", type=int, default=128, help="size of image after scaled")
parser.add_argument("--imageSize", type=int, default=64, help="size of each image dimension")
parser.add_argument("--nc", type=int, default=3, help="number of image channels")
parser.add_argument("--ngf", type=int, default=64, help="channels of middle layers for generator")
parser.add_argument("--lambda_score", type=float, default=0.9, help="weight of reconstruction error")
parser.add_argument("--gen_pth", default=r"../experiments/skip_GANomaly_train/gen.pth", help="pretrained model of gen")
parser.add_argument("--disc_pth", default=r"../experiments/skip_GANomaly_train/disc.pth", help="pretrained model of disc")
opt = parser.parse_args()
print(opt)
os.makedirs(opt.experiment, exist_ok=True)

## random seed
# opt.seed = 42
# torch.manual_seed(opt.seed)
# np.random.seed(opt.seed)

## cudnn
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

## device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

## dataset
test_dataset = load_dataset(opt.dataroot, opt.dataset, opt.size, trans=None, train=False)
test_dataloader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False)
opt.dataSize = test_dataset.__len__()

## model
gen = NetG(opt).to(device)
disc = NetD(opt).to(device)
assert gen.load_state_dict(torch.load(opt.gen_pth))
assert disc.load_state_dict(torch.load(opt.disc_pth))
print("Pretrained models have been loaded.")

## record results
writer = SummaryWriter("../runs{0}".format(opt.experiment[1:]), comment=opt.experiment[1:])

## def
def splitImage(img, size):
    if img.size(3) % size !=0:
        return
    num = int(img.size(3) / size)
    results = torch.zeros(num**2, img.size(1), size, size)
    split1 = torch.split(img, size, dim=2)
    for i in range(num):
        split2 = torch.split(split1[i], size, dim=3)
        for j in range(num):
            results[i*num+j, :, :, :] = split2[j]
    return results


def catImage(imgs, size):
    if imgs.size(0) != size[0] * size[1]:
        return
    results = torch.zeros(1, imgs.size(1), imgs.size(2)*size[0], imgs.size(3)*size[1])
    width = imgs.size(2)
    height = imgs.size(3)
    for i in range(size[0]):
        for j in range(size[1]):
            results[0, :, i*width:(i+1)*width, j*height:(j+1)*height] = imgs[i*size[0]+j]
    return results

## loss
L_con = nn.L1Loss(reduction='mean')
L_lat = nn.MSELoss(reduction='mean')

## test
gen.eval()
disc.eval()
con_loss = []
lat_loss = []
total_loss = []
labels = []
evaluation = Evaluate(opt.experiment)
tqdm_loader = tqdm.tqdm(test_dataloader)
for i, (test_input, label, mask) in enumerate(tqdm_loader):
    tqdm_loader.set_description(f"Test Sample {i+1} / {opt.dataSize}")
    patches_num = int(test_input.size(3) / opt.imageSize)
    test_inputs = splitImage(test_input, opt.imageSize).to(device)
    ## inference
    with torch.no_grad():
        outputs = gen(test_inputs)
        _, features_real = disc(test_inputs)
        _, features_feak = disc(outputs)

    con = L_con(outputs, test_inputs).item()
    lat = L_lat(features_real, features_feak).item()
    total_loss.append(con*opt.lambda_score + lat)
    labels.append(label)

    output = catImage(outputs, (patches_num, patches_num))
    residule = torch.abs(test_input - output)

    vutils.save_image(torch.cat((test_input, output, residule), dim=0), '{0}/{1}-0.png'.format(opt.experiment, i))
    vutils.save_image(mask, '{0}/{1}-1.png'.format(opt.experiment, i))

    residule = residule.detach().cpu().numpy()
    residule = draw_heatmap(residule)
    cv2.imwrite('{0}/{1}-2.png'.format(opt.experiment, i), residule)

total_loss = np.array(total_loss)
total_loss = (total_loss - np.min(total_loss)) / (np.max(total_loss) - np.min(total_loss))
evaluation.labels = labels
evaluation.scores = total_loss
evaluation.run()