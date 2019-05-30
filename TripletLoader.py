from torch.utils.data.dataset import Dataset
import os
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import torch
from torchvision.utils import save_image


class TripletLoader(Dataset):

    def __init__(self, root, transform=None, target_transform=None):
        super(TripletLoader, self).__init__()
        self.data_path = root
        self.transform = transform
        self.imgs = os.listdir(root)
        self.imgs = [el for el in self.imgs if os.path.splitext(el)[1] == '.jpg']
        self.labels = [int(el.split('_')[0]) for el in self.imgs]
        self.classes = np.unique(self.labels)
        self.camids = [int(el.split('_')[1][1]) for el in self.imgs]
        self.imgs = [os.path.join(root, el) for el in self.imgs]

    def __getitem__(self, index):
        image = self.imgs[index]
        label = self.labels[index]
        neg_lbl = np.random.choice([i for i in self.labels if i != self.labels[index]])

        indices, = np.where(self.labels == neg_lbl)

        neg_idx = np.random.choice(indices)
        neg_img = self.imgs[neg_idx]
        indices, = np.where(self.labels == np.int32(label))

        pos_idx = np.random.choice(indices[np.where(indices != index)])

        pos_img = self.imgs[pos_idx]

        image = Image.open(image)
        pos_img = Image.open(pos_img)
        neg_img = Image.open(neg_img)

        image = image.convert('RGB')
        neg_img = neg_img.convert('RGB')
        pos_img = pos_img.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
            neg_img = self.transform(neg_img)
            pos_img = self.transform(pos_img)

        assert self.labels[index] != neg_lbl
        assert self.labels[pos_idx] == label
        return image, pos_img, label, neg_img, neg_lbl

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((160, 128), interpolation=3),
        transforms.ToTensor()
    ])

    data_dir = '/home/paul/datasets/market1501/bounding_box_train'
    dset_train = TripletLoader(data_dir, transform=transform)
    train_loader = DataLoader(dset_train, batch_size=4,
                              shuffle=True, num_workers=1)
    classes = dset_train.classes
    #print(len(classes))
    train_iter = iter(train_loader)
    batch = next(train_iter)
    inputs, pos, targets, negs, neg_targets = batch
    #inputs = np.stack(inputs, axis=1)
    #inputs = torch.from_numpy(inputs)
    #inputs = inputs.transpose(0, 1)
    #print(inputs.shape)
    #print(neg.shape)
    print(targets)
    print(neg_targets)
    save_image(inputs, 'images/inputs.png')
    save_image(pos, 'images/pos.png')
    save_image(negs, 'images/neg.png')
    save_image(inputs[0], 'images/loader[0].png')
