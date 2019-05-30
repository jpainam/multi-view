import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.models import ResNet
from loss import MultiViewSimilarityLoss
import torchvision.transforms as transforms
from model import Model
import argparse
import numpy as np
import time
from TripletLoader import TripletLoader

# from logger import Logger
import util
from MultiViewDataLoader import MultiViewDataSet


parser = argparse.ArgumentParser(description='Train Triplet')
parser.add_argument('--data', metavar='DIR', help='path to dataset',
                    default='/home/fstu1/datasets/market1501/bounding_box_train')

parser.add_argument('--epochs', default=150, type=int, metavar='N', help='number of total epochs to run (default: 100)')
parser.add_argument('-b', '--batch_size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 4)')
parser.add_argument('--gpu', default=1, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate (default: 0.0001)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--lr-decay-freq', default=30, type=float,
                    metavar='W', help='learning rate decay (default: 30)')
parser.add_argument('--lr-decay', default=0.1, type=float,
                    metavar='W', help='learning rate decay (default: 0.1)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', dest='resume', action='store_true', help='use checkpoint ')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')

args = parser.parse_args()

print('Loading data')

transform = transforms.Compose([
    transforms.Resize((256, 128), interpolation=3),
    #transforms.RandomCrop(224, 224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

device = torch.device("cuda:{}".format(torch.cuda.current_device()))

# Load dataset
dset_train = TripletLoader(args.data, transform=transform)
train_loader = DataLoader(dset_train, batch_size=args.batch_size,
                          shuffle=True, num_workers=16)

classes = dset_train.classes

model = Model(num_classes=len(classes), training=True)

model.to(device)
cudnn.benchmark = True

print('Running on ' + str(device))

# logger = Logger('logs')

# Loss and Optimizer
lr = args.lr
n_epochs = args.epochs
criterion = nn.CrossEntropyLoss()
triplet_criterion = nn.TripletMarginLoss(margin=0.3)
#optimizer = torch.optim.SGD([
#    {'params': model.base.parameters(), 'lr': 0.001},
#    {'params': model.classifier.parameters(), 'lr': 0.01}
#], weight_decay=5e-4, momentum=0.9, nesterov=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Decay LR by a factor of 0.1 every 40 epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

best_acc = 0.0
best_loss = 0.0
start_epoch = 0


# Helper functions
def load_checkpoint():
    global best_acc, start_epoch
    # Load checkpoint.
    print('\n==> Loading checkpoint..')
    #assert os.path.isfile(args.resume), 'Error: no checkpoint file found!'
    checkpoint_path = './checkpoint/triplet_checkpoint.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    #best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def train():
    train_size = len(train_loader)

    for i, (inputs, pos, targets, negs, neg_targets) in enumerate(train_loader):
        # Convert from list of 3D to 4D

        #inputs = np.stack(inputs, axis=1)
        current_batch_size, c, h, w = inputs.shape
        if current_batch_size < args.batch_size:    # skip the last batch
            continue
        #inputs = torch.from_numpy(inputs)

        inputs, pos = inputs.cuda(device), pos.cuda(device)
        targets, negs = targets.cuda(device), negs.cuda(device)
        inputs, targets, negs = Variable(inputs), Variable(targets), Variable(negs)
        pos = Variable(pos)
        # compute output
        anchor_f = model(inputs)
        pos_f = model(pos)
        neg_f = model(negs)

        # loss = criterion(outputs, targets)
        #sim_loss = similarity_criterion(embbedings, negative)
        loss = triplet_criterion(anchor_f, pos_f, neg_f)
        #loss = loss + 0.1*sim_loss
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % args.print_freq == 0:
            print("\tIter [%d/%d] Loss: %.4f" % (i + 1, train_size, loss.item()))


# Validation and Testing
def eval(data_loader, is_test=False):
    if is_test:
        load_checkpoint()

    # Eval
    total = 0.0
    correct = 0.0

    total_loss = 0.0
    n = 0

    for i, (inputs, targets) in enumerate(data_loader):
        with torch.no_grad():
            # Convert from list of 3D to 4D
            inputs = np.stack(inputs, axis=1)

            inputs = torch.from_numpy(inputs)

            inputs, targets = inputs.cuda(device), targets.cuda(device)
            inputs, targets = Variable(inputs), Variable(targets)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss
            n += 1

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted.cpu() == targets.cpu()).sum()

    avg_test_acc = 100 * correct / total
    avg_loss = total_loss / n

    return avg_test_acc, avg_loss


if __name__ == '__main__':
    # Training / Eval loop
    if args.resume:
        load_checkpoint()

    for epoch in range(start_epoch, n_epochs):
        print('\n-----------------------------------')
        print('Epoch: [%d/%d]' % (epoch+1, n_epochs))
        start = time.time()

        model.train()
        scheduler.step()
        train()
        print('Time taken: %.2f sec.' % (time.time() - start))

        util.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, 'triplet')

        # Decaying Learning Rate
        if (epoch + 1) % args.lr_decay_freq == 0:
            lr *= args.lr_decay
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            print('Learning rate:', lr)
