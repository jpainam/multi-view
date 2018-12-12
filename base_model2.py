import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MultiViewModel(nn.Module):
    def __init__(self, n_classe, n_view):
        super(MultiViewModel, self).__init__()
        self.n_classe = n_classe
        self.n_view = n_view

        self.conv1 = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        ) for _ in range(n_view)])

        self.maxpool = nn.MaxPool2d(stride=(2, 2), kernel_size=(3, 3), padding=1)
        self.conv2 = nn.ModuleList([nn.Conv2d(
            in_channels=64, out_channels=64,
            kernel_size=(1, 1), stride=(1, 1))
            for _ in range(n_view)])

        self.conv3 = nn.ModuleList([nn.Conv2d(
            in_channels=64, out_channels=128,
            kernel_size=(3, 3), stride=(2, 2))
            for _ in range(n_view)])

        self.conv4 = nn.ModuleList([nn.Conv2d(
            in_channels=128, out_channels=256,
            kernel_size=(3, 3), stride=(2, 2))
            for _ in range(n_view)])

        self.conv5 = nn.ModuleList([nn.Conv2d(
            in_channels=256, out_channels=512,
            kernel_size=(3, 3), stride=(2, 2))
            for _ in range(n_view)])

        self.fc1 = nn.Linear(512 * n_view, 1024)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, n_classe)

    def forward(self, x):
        for i, conv in enumerate(self.conv1):
            x[i] = conv()
        exit(0)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    model = MultiViewModel(n_classe=750, n_view=4)
    batch = Variable(torch.randn(4, 1, 3, 224, 224)) # n_views x b x c x h x w
    model = model.to('cuda:0')
    batch = batch.to('cuda:0')
    predict = model(batch)
    print(predict)