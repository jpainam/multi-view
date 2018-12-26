import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision.utils import save_image
from base_model import *


######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x


class Model(nn.Module):
    def __init__(self, num_classes=None, training=False):
        super(Model, self).__init__()
        self.base = resnet50(pretrained=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = ClassBlock(2048, num_classes)
        self.training = training

    def forward(self, xx, neg=None):
        # Training

        # xx shape [View, B, C, H, W]
        # neg shape [B, C, H, W]
        if not self.training:
            v = self.base(xx)
            v = self.avgpool(v)
            v = v.view(v.size(0), -1)
            v = self.classifier(v)
            return v
        else:
            xx = xx.transpose(0, 1)
            print(xx.size())
            combined_views = []
            for v in xx:
                v = self.base(v)
                v = self.avgpool(v)
                v = v.view(v.size(0), -1)
                combined_views.append(v)
            print(len(combined_views))
            print(torch.stack(combined_views))
            exit(0)
            pooled_view = torch.max(torch.stack(combined_views))
            #pooled_view = combined_views[0]
            #for i in range(1, len(combined_views)):
            #    pooled_view = torch.max(pooled_view, combined_views[i])
            pooled_view = self.classifier(pooled_view)
            #neg = self.base(neg)
            #neg = self.avgpool(neg)
            #neg = neg.view(neg.size(0), -1)
            return pooled_view
