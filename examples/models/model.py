import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(5 * 5 * 64, 10)

    def forward(self, x):
        x = self.block(x)
        x = x.view(-1, 5 * 5 * 64)
        x = self.fc(x)
        return x


class ModelBN(nn.Module):
    def __init__(self):
        super(ModelBN, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(5 * 5 * 64, 10)

    def forward(self, x):
        x = self.block(x)
        x = x.view(-1, 5 * 5 * 64)
        x = self.fc(x)
        return x


class ModelBNNoReLU(nn.Module):
    def __init__(self):
        super(ModelBNNoReLU, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(5 * 5 * 64, 10)

    def forward(self, x):
        x = self.block(x)
        x = x.view(-1, 5 * 5 * 64)
        x = self.fc(x)
        return x


class ModelLinear(nn.Module):
    def __init__(self,):
        super(ModelLinear, self).__init__()
        self.linear1 = nn.Linear(28*28, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, 10)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        return x


class ModelShortCut(nn.Module):
    def __init__(self):
        super(ModelShortCut, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(2)
        self.linear = nn.Linear(32*6*6, 10)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x) + x
        x = self.maxpool(x)
        x = x.view(-1, 32*6*6)
        x = self.linear(x)
        return x

    
class ModelLayerNorm(nn.Module):
    def __init__(self):
        super(ModelLayerNorm, self).__init__()
        self.block1 = nn.Sequential(
            nn.Linear(784, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        self.linear = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.block1(x)
        x = self.block2(x)
        x = self.linear(x)
        return x

