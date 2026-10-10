"""Residual convolutional rainfall-runoff forecast network."""

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.leakyrelu = nn.LeakyReLU(0.2,inplace=True)
    def forward(self, x):
        y = self.leakyrelu(self.conv1(x))
        y = self.conv2(y)
        return self.leakyrelu(x + y)


class cnnret(nn.Module):
    def __init__(self, jiedianshu,num):
        super(cnnret, self).__init__()
        self.out6 = nn.Sequential(
                    nn.Conv1d(2,16,kernel_size=3,stride=1),
                    nn.BatchNorm1d(16),
                    nn.LeakyReLU(0.2,inplace=True),
                    nn.Conv1d(16, 64, kernel_size=2, stride=1),
                    nn.BatchNorm1d(64),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv1d(64, jiedianshu, kernel_size=3, padding=1),
                    nn.BatchNorm1d(jiedianshu),
                    nn.LeakyReLU(0.2, inplace=True),
                    ResidualBlock(jiedianshu),
                    ResidualBlock(jiedianshu),
                    )
        self.out12 = nn.Sequential(
                    nn.Conv1d(2,16,kernel_size=3,stride=2),
                    nn.BatchNorm1d(16),
                    nn.LeakyReLU(0.2,inplace=True),
                    nn.Conv1d(16, 64, kernel_size=3, stride=1),
                    nn.BatchNorm1d(64),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv1d(64, jiedianshu, kernel_size=3, padding=1),
                    nn.BatchNorm1d(jiedianshu),
                    nn.LeakyReLU(0.2, inplace=True),
                    ResidualBlock(jiedianshu),
                    ResidualBlock(jiedianshu),
                    )
        self.out24 = nn.Sequential(
                    nn.Conv1d(2,16,kernel_size=3,stride=2),
                    nn.BatchNorm1d(16),
                    nn.LeakyReLU(0.2,inplace=True),
                    nn.Conv1d(16, 64, kernel_size=3, stride=2),
                    nn.BatchNorm1d(64),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv1d(64, jiedianshu, kernel_size=3, stride=1),
                    nn.BatchNorm1d(jiedianshu),
                    nn.LeakyReLU(0.2, inplace=True),
                    ResidualBlock(jiedianshu),
                    ResidualBlock(jiedianshu),
                    )
        self.rainout6 = nn.Sequential(
                    nn.Conv1d(1,16,kernel_size=3,stride=1),
                    nn.BatchNorm1d(16),
                    nn.LeakyReLU(0.2,inplace=True),
                    nn.Conv1d(16, 64, kernel_size=2, stride=1),
                    nn.BatchNorm1d(64),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv1d(64, jiedianshu, kernel_size=3, padding=1),
                    nn.BatchNorm1d(jiedianshu),
                    nn.LeakyReLU(0.2, inplace=True),
                    ResidualBlock(jiedianshu),
                    ResidualBlock(jiedianshu),
                    )
        self.rainout12 = nn.Sequential(
                    nn.Conv1d(1,16,kernel_size=3,stride=2),
                    nn.BatchNorm1d(16),
                    nn.LeakyReLU(0.2,inplace=True),
                    nn.Conv1d(16, 64, kernel_size=3, stride=1),
                    nn.BatchNorm1d(64),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv1d(64, jiedianshu, kernel_size=3, padding=1),
                    nn.BatchNorm1d(jiedianshu),
                    nn.LeakyReLU(0.2, inplace=True),
                    ResidualBlock(jiedianshu),
                    ResidualBlock(jiedianshu),
                    )
        self.rainout24 = nn.Sequential(
                    nn.Conv1d(1,16,kernel_size=3,stride=2),
                    nn.BatchNorm1d(16),
                    nn.LeakyReLU(0.2,inplace=True),
                    nn.Conv1d(16, 64, kernel_size=3, stride=2),
                    nn.BatchNorm1d(64),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv1d(64, jiedianshu, kernel_size=3, stride=1),
                    nn.BatchNorm1d(jiedianshu),
                    nn.LeakyReLU(0.2, inplace=True),
                    ResidualBlock(jiedianshu),
                    ResidualBlock(jiedianshu),
                    )
        self.yubao = nn.Sequential(
                    nn.ConvTranspose1d(jiedianshu*2,256,kernel_size=4,stride=2),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(0.2,inplace=True),
                    nn.ConvTranspose1d(256,64,kernel_size=4,stride=2),
                    nn.BatchNorm1d(64),
                    nn.LeakyReLU(0.2,inplace=True),
                    nn.ConvTranspose1d(64, 16, kernel_size=4, stride=1),
                    nn.BatchNorm1d(16),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.ConvTranspose1d(16,1,kernel_size=4,stride=1),
                    nn.BatchNorm1d(1),
                    nn.LeakyReLU(0.2,inplace=True),
                    )

    def forward(self, x, y, pre, num):
        # help = self.SelfAttention(x.permute(0, 2, 1)).permute(0, 2, 1)
        if pre == 6:
            preout = self.out6(x)
        elif pre ==12:
            preout = self.out12(x)
        elif num ==24:
            preout = self.out24(x)
        if num == 6:
            rainout = self.rainout6(y)
        elif num ==12:
            rainout = self.rainout12(y)
        elif num ==24:
            rainout = self.rainout24(y)
        zuhe = torch.cat([preout, rainout], 1)
        out = self.yubao(zuhe).squeeze(1)
        return out


RCNN = cnnret
