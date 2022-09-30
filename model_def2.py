import torch
from torch.nn import Softmax, ReLU, Sequential, Conv3d, MaxPool2d, BatchNorm2d, Dropout, Linear, Conv2d,BatchNorm1d,LayerNorm
import torch.nn as nn
import torch.nn.functional as F
class ArcFaceNet(nn.Module):
    def __init__(self, cls_num=10, feature_dim=2):
        super(ArcFaceNet, self).__init__()
        self.w = nn.Parameter(torch.randn(feature_dim, cls_num))
    def forward(self, features, m=1, s=10):
        _features = nn.functional.normalize(features, dim=1)
        _w = nn.functional.normalize(self.w, dim=0)

        theta = torch.acos(torch.matmul(_features, _w) / 10)
        numerator = torch.exp(s * torch.cos(theta + m))
        denominator = torch.sum(torch.exp(s * torch.cos(theta)), dim=1, keepdim=True) - torch.exp(
            s * torch.cos(theta)) + numerator
        return torch.log(torch.div(numerator, denominator))
class CenterLossNet(nn.Module):
    def __init__(self, cls_num=10, feature_dim=2):
        super(CenterLossNet, self).__init__()
        self.centers = nn.Parameter(torch.randn(cls_num, feature_dim))

    def forward(self, features, labels, reduction='mean'):
        _features = nn.functional.normalize(features)
        centers_batch = self.centers.index_select(dim=0, index=labels.long())
        if reduction == 'sum':
            return torch.sum(torch.pow(_features - centers_batch, 2)) / 2
        elif reduction == 'mean':
            return torch.sum(torch.pow(_features - centers_batch, 2)) / 2 / len(features)
        else:
            raise ValueError("ValueError: {0} is not a valid value for reduction".format(reduction))
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        # return self.relu(x)
        return self.sigmoid(x)
class ArcCNN(nn.Module):
    def __init__(self,device):
        super(ArcCNN, self).__init__()
        self.feature_extractor1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=60,
                kernel_size=(1,40),
                stride=1,
            ),
            nn.BatchNorm2d(60),
            nn.ReLU(),
        )
        self.feature_extractor2 = nn.Sequential(
            nn.Conv1d(60,60,(22,1),1),
            nn.BatchNorm2d(60),
            nn.ReLU(),
            nn.Dropout(0.8)
        )
        self.feature_extractor3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1,75), stride=15),
            nn.BatchNorm2d(60),
            nn.ReLU(),
            nn.Dropout(0.8)
        )
        self.linear = nn.Sequential(
            nn.Linear(60*5*5,1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256,128),
        )

        self.FC = Sequential(
            Linear(128, 4),
            nn.LogSoftmax(dim=1)
        )

        self.classifier = Sequential(
            Linear(30, 30),
            nn.BatchNorm1d(30),
            nn.ReLU(),
            Linear(30,4),
            nn.BatchNorm1d(4),
            nn.Softmax()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.inplanes = 22
        self.ca = ChannelAttention(self.inplanes)
        self.ca2 = ChannelAttention(60)
        self.sa = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.softmax = nn.LogSoftmax()

    def forward(self, x ):

        x = x.reshape((len(x),22,1,1000))
        x = self.ca(x) * x
        x = self.sa(x) * x
        x = x.reshape((len(x),1,22,1000))
        x = self.feature_extractor1(x)
        x = self.feature_extractor2(x)
        x = self.feature_extractor3(x)
        x = self.avgpool(x)
        features = self.linear(x.view(x.size(0), -1))
        out = self.FC(features)
        return features,out