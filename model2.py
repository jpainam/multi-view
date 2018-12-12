import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import ConvLayer, AvgPooling, MaxPooling
from collections import OrderedDict

class MultiView(nn.Module):
    def __init__(self, n_classe):
        super(MultiView, self).__init__()
        # Layer 1
        self.layer1 = nn.Sequential(OrderedDict([
          ('conv1', ConvLayer(3, out_channels=32, filter_size=(3,3), stride=(2,2)))
        ]))
        # Layer 2
        self.layer2 = nn.Sequential(OrderedDict([
            ('conv2a', ConvLayer(32, out_channels=64, filter_size=(3, 3), stride=(2, 2))),
            ('conv2b', ConvLayer(64, out_channels=64, filter_size=(3, 3), stride=(1, 1))),
            ('conv2c', ConvLayer(64, out_channels=64, filter_size=(3, 3), stride=(1, 1))),
        ]))
        # Layer 3
        self.layer3 = nn.Sequential(OrderedDict([
            ('conv3a', ConvLayer(64, out_channels=128, filter_size=(3, 3), stride=(1, 1))),
            ('conv3b', ConvLayer(128, out_channels=128, filter_size=(3, 3), stride=(1, 1))),
            ('conv3c', ConvLayer(128, out_channels=128, filter_size=(3, 3), stride=(1, 1))),
        ]))
        # Layer 4
        self.layer4 = nn.Sequential(OrderedDict([
            ('conv4a', ConvLayer(128, out_channels=256, filter_size=(3, 3), stride=(1, 1))),
            ('conv4b', ConvLayer(256, out_channels=256, filter_size=(3, 3), stride=(1, 1))),
            ('conv4c', ConvLayer(256, out_channels=256, filter_size=(3, 3), stride=(1, 1))),
        ]))

        # Layer 5
        self.layer5 = nn.Sequential(OrderedDict([
            ('conv5a', ConvLayer(256, out_channels=512, filter_size=(3, 3), stride=(1, 1))),
            ('conv5b', ConvLayer(512, out_channels=512, filter_size=(3, 3), stride=(1, 1))),
            ('conv5c', ConvLayer(512, out_channels=512, filter_size=(3, 3), stride=(1, 1))),
        ]))
        # FC
        self.max_pool = MaxPooling()
        self.avg_pool = AvgPooling()
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(512 * 2, 512 * 2)
        self.fc2 = nn.Linear(512 * 2, n_classe)

    def forward(self, x):
        x = self.layer1(x)

        x = self.avg_pool(x)
        x = torch.cat([
            x['view_1'],
            x['view_2']
        ], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
