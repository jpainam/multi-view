import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size=(3, 3), stride=(1, 1)):
        super(ConvLayer, self).__init__()
        self.odd = nn.Sequential(
            nn.Conv2d( in_channels=in_channels, out_channels=out_channels, filter_size=filter_size, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.even = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, filter_size=filter_size, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return {
            'view_1': self.odd(x['view_1']),
            'view_2': self.even(x['view_2'])
        }


class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, x,kernel_size=(2,2)):
        return {
            'view_1': F.max_pool2d(x['view_1'], kernel_size=kernel_size),
            'view_2': F.max_pool2d(x['view_2'], kernel_size=kernel_size)
        }


class AvgPooling(nn.Module):
    def __init__(self):
        super(AvgPooling, self).__init__()

    def forward(self, x):
        n, c, h, w = x['view_1'].size()
        view_1 = x['view_1'].view(n, c, -1).mean(-1)
        view_2 = x['view_2'].view(n, c, -1).mean(-1)
        return {
            'view_1': view_1,
            'view_2': view_2
        }
