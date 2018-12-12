from torch.utils.data.dataset import Dataset
import os
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import torch
from torchvision.utils import save_image


class MultiViewDataSet(Dataset):

    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx

    def __init__(self, root, transform=None, target_transform=None):
        self.x = []
        self.y = []
        self.root = root

        self.classes, self.class_to_idx = self.find_classes(root)

        self.transform = transform
        self.target_transform = target_transform

        # root / <label>  / <train/test> / <item> / <view>.png
        for label in os.listdir(root): # Label
            for item in os.listdir(root + '/' + label):
                views = []
                for view in os.listdir(root + '/' + label + '/' + item):
                    views.append(root + '/' + label + '/' + item + '/' + view)

                self.x.append(views)
                self.y.append(self.class_to_idx[label])

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        orginal_views = self.x[index]
        views = []
        neg_idx = np.random.choice([i for i in self.y if i != self.y[index]])
        neg_view = np.random.choice(self.x[neg_idx])

        for view in orginal_views:
            im = Image.open(view)
            im = im.convert('RGB')
            if self.transform is not None:
                im = self.transform(im)
            views.append(im)
        im_neg = self.transform(Image.open(neg_view).convert('RGB'))

        assert self.y[index] != neg_idx
        return views, self.y[index], im_neg, neg_idx

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.x)


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((160, 128), interpolation=3),
        transforms.ToTensor()
    ])

    data_dir = '/home/paul/datasets/market1501/multiviews'
    dset_train = MultiViewDataSet(data_dir, transform=transform)
    train_loader = DataLoader(dset_train, batch_size=4,
                              shuffle=True, num_workers=1)
    classes = dset_train.classes
    print(len(classes))
    train_iter = iter(train_loader)
    batch = next(train_iter)
    inputs, target, neg, neg_target = batch
    inputs = np.stack(inputs, axis=1)
    inputs = torch.from_numpy(inputs)
    inputs = inputs.transpose(0, 1)
    print(inputs.shape)
    print(neg.shape)
    print(target)
    print(neg_target)
    save_image(neg, 'images/neg.png')
    save_image(inputs[0], 'images/loader[0].png')
