# -*- coding:utf8 -*-
# @TIME     : 2020/12/11 15:32
# @Author   : SuHao
# @File     : DCGAN_skip_GANomaly.py


import torch.nn as nn
import torch
from torchsummary import summary

##
class NetG(nn.Module):
    """
    GENERATOR NETWORK
    """

    def __init__(self, opt):
        super(NetG, self).__init__()
        ngf = 64
        self.e1 = nn.Sequential(
            nn.Conv2d(opt.nc, ngf, 4, 2, 1, bias=False),            # 32*32
        )
        self.e2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, ngf << 1, 4, 2, 1, bias=False),          # 16*16
            nn.BatchNorm2d(ngf << 1),       # 128
        )
        self.e3 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf << 1, ngf << 2, 4, 2, 1, bias=False),     # 8*8
            nn.BatchNorm2d(ngf << 2),       # 256
        )
        self.e4 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf << 2, ngf << 3, 4, 2, 1, bias=False),     # 4*4
            nn.BatchNorm2d(ngf << 3),       # 512
        )
        self.e5 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf << 3, ngf << 3, 4, 1, 0, bias=False),       # 1*1
            nn.BatchNorm2d(ngf << 3),
        )

        self.d5 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf << 3, ngf << 3, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf << 3),
        )
        self.d4 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf << 4, ngf << 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf << 2),
        )
        self.d3 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf << 2, ngf << 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf << 1),
        )
        self.d2 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf << 1, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
        )
        self.d1 = nn.Sequential(
            nn.ConvTranspose2d(ngf, opt.nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        e1_out = self.e1(x)
        e2_out = self.e2(e1_out)
        e3_out = self.e3(e2_out)
        e4_out = self.e4(e3_out)
        e5_out = self.e5(e4_out)
        d5_out = self.d5(e5_out)
        d4_out = self.d4(torch.cat([e4_out, d5_out], dim=1))
        # d4_out = self.d4(d5_out)
        # d3_out = self.d3(torch.cat([e3_out, d4_out], dim=1))
        d3_out = self.d3(d4_out)
        # d2_out = self.d2(torch.cat([e2_out, d3_out], dim=1))
        d2_out = self.d2(d3_out)
        # gen_imag = self.d1(torch.cat([e1_out, d2_out], dim=1))
        gen_img = self.d1(d2_out)
        return gen_img

##
class NetD(nn.Module):
    def __init__(self, opt):
        super(NetD, self).__init__()
        ngf = 64
        self.model = nn.Sequential(
            nn.Conv2d(opt.nc, ngf, 4, 2, 1, bias=False),

            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, ngf << 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf << 1),       # 128

            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf << 1, ngf << 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf << 2),       # 256

            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf << 2, ngf << 3, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf << 3),       # 512

            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf << 3, 100, 4, 1, 0, bias=False),       # 512
            nn.BatchNorm2d(100),       # 100
        )

        self.classify = nn.Sequential(
            nn.Conv2d(100, 1, 3, 1, 1, bias=False),       # 512
            nn.Sigmoid(),
        )


    def forward(self, x):
        feature = self.model(x)
        classification = self.classify(feature)
        return classification.view(-1, 1).squeeze(1), feature


def print_net():
    class OPT:
        def __init__(self, imageSize, nz, nc, ngf, ngpu, n_extra_layers):
            self.imageSize = imageSize
            self.nz = nz
            self.nc = nc
            self.ngf = ngf
            self.ngpu = ngpu
            self.n_extra_layers = n_extra_layers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = OPT(64, 100, 3, 64, 1, 0)
    gen = NetG(opt).to(device)
    opt = OPT(64, 1, 3, 64, 1, 0)
    disc = NetD(opt).to(device)
    summary(gen, (3, 64, 64))
    summary(disc, (3, 64, 64))

print_net()