import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from base_model import *


class Model(nn.Module):
    def __init__(self, num_classes=None):
        super(Model, self).__init__()
        self.base = resnet50(pretrained=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        planes = 2048
        if num_classes is not None:
            self.fc = nn.Linear(planes, num_classes)
            init.normal_(self.fc.weight, std=0.001)
            init.constant_(self.fc.bias, 0)

    def forward(self, xx, neg):
        # xx shape [View, B, C, H, W]
        # neg shape [B, C, H, W]

        xx = xx.transpose(0, 1)
        combined_views = []
        for v in xx:
            v = self.base(v)
            v = self.avgpool(v)
            v = v.view(v.size(0), -1)
            combined_views.append(v)

        pooled_view = combined_views[0]
        for i in range(1, len(combined_views)):
            pooled_view = torch.max(pooled_view, combined_views[i])
        pooled_view = self.fc(pooled_view)
        neg = self.base(neg)
        neg = self.avgpool(neg)
        neg = neg.view(neg.size(0), -1)
        return pooled_view, combined_views, neg
