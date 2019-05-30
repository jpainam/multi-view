import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.models import ResNet
from loss import MultiViewSimilarityLoss
import torchvision.transforms as transforms
from model import Model, SimilarityModel
from random_erasing import RandomErasing
import argparse
from SimilarityLoss import SimilarityLoss
import numpy as np
import time

# from logger import Logger
import util
from MultiViewDataLoader import MultiViewDataSet

parser = argparse.ArgumentParser(description='MV-CNN4ReID')
parser.add_argument('--data', metavar='DIR', help='path to dataset',
                    default='/home/fstu1/datasets/market1501/multiviews')

parser.add_argument('--epochs', default=150, type=int, metavar='N', help='number of total epochs to run (default: 100)')
parser.add_argument('-b', '--batch_size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 4)')
parser.add_argument('--gpu', default=1, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate (default: 0.0001)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training')
parser.add_argument('--lr-decay-freq', default=30, type=float,
                    metavar='W', help='learning rate decay (default: 30)')
parser.add_argument('--lr-decay', default=0.1, type=float,
                    metavar='W', help='learning rate decay (default: 0.1)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', dest='resume', action='store_true', help='use checkpoint ')
parser.add_argument('--erasing_p', default=0.8, type=float,
                    help='Random Erasing probability, in [0,1]')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')

args = parser.parse_args()

print('Loading data')

transform_list = [
    transforms.Resize((256, 128), interpolation=3),
    transforms.Pad(10),
    transforms.RandomCrop((256, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]
if args.erasing_p > 0:
    transform_list.append(RandomErasing(probability=args.erasing_p, mean=[0.0, 0.0, 0.0]))

if args.color_jitter:
    transform_list.append(transforms.ColorJitter(brightness=0.1, contrast=0.1,
                                                   saturation=0.1, hue=0))
# Load dataset
transform_list = transforms.Compose(transform_list)
dset_train = MultiViewDataSet(args.data, transform=transform_list)
train_loader = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=8, pin_memory=True)

classes = dset_train.classes
# print(len(classes))

model = Model(class_num=len(classes), stride=2)

model = model.cuda()
cudnn.benchmark = True
sim_model = SimilarityModel()
sim_model = sim_model.cuda()
classifier = model.classifier.classifier
# logger = Logger('logs')

# Loss and Optimizer
lr = args.lr
n_epochs = args.epochs
criterion = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss()
similarity_criterion = SimilarityLoss()

ignored_params = list(map(id, model.classifier.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
optimizer = torch.optim.SGD([
    {'params': base_params, 'lr': 0.001},
    {'params': model.classifier.parameters(), 'lr': 0.01}
], weight_decay=5e-4, momentum=0.9, nesterov=True)

sim_optim = torch.optim.SGD([
    {'params': sim_model.fc.parameters(), 'lr': 0.01}
], weight_decay=5e-4, momentum=0.9, nesterov=True)

# Decay LR by a factor of 0.1 every 40 epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
scheduler_sim = torch.optim.lr_scheduler.StepLR(sim_optim, step_size=40, gamma=0.1)

best_acc = 0.0
best_loss = 0.0
start_epoch = 0


# Helper functions
def load_checkpoint():
    global best_acc, start_epoch
    # Load checkpoint.
    print('\n==> Loading checkpoint..')
    # assert os.path.isfile(args.resume), 'Error: no checkpoint file found!'
    checkpoint_path = './checkpoint/mvcnn_checkpoint.pth'
    checkpoint = torch.load(checkpoint_path)
    # best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

from collections import defaultdict
pos_loss = defaultdict(list)
neg_loss = defaultdict(list)
id_loss = defaultdict(list)

def train(epoch):
    train_size = len(train_loader)

    for i, (inputs, targets, negs, neg_target) in enumerate(train_loader):
        # Convert from list of 3D to 4D
        optimizer.zero_grad()
        sim_optim.zero_grad()
        inputs = np.stack(inputs, axis=1)
        inputs = torch.from_numpy(inputs)
        inputs = inputs.transpose(0, 1)
        inputs, targets, negs = inputs.cuda(), targets.cuda(), negs.cuda()
        neg_target = neg_target.cuda()
        inputs, targets, negs = Variable(inputs), Variable(targets), Variable(negs)
        neg_target = Variable(neg_target)
        aggregate_views = []
        view_loss = 0.0

        for v in inputs:
            current_batch_size, c, h, w = v.shape
            if current_batch_size < args.batch_size:  # skip the last batch
                continue
            outputs, f = model(v)
            loss = criterion(outputs, targets)
            loss.backward(retain_graph=True)
            view_loss = view_loss + loss.item()
            aggregate_views.append(f)
        if (i + 1) % args.print_freq == 0:
            print("\t[%d] View Loss: %.4f" % (len(inputs), view_loss/len(inputs)))
            id_loss[epoch].append(view_loss/len(inputs))
        if len(aggregate_views) == 0:
            continue
        pf = torch.mean(torch.stack(aggregate_views), dim=0)
        pos_outputs = classifier(pf)
        loss = criterion(pos_outputs, targets)
        loss.backward(retain_graph=True)
        if (i + 1) % args.print_freq == 0:
            print("Aggregate Loss: %.4f" % loss.item())
            pos_loss[epoch].append(loss.item())

        neg_outputs, nf = model(negs)
        # optimizer.zero_grad()
        loss = criterion(neg_outputs, neg_target)
        loss.backward()
        if (i + 1) % args.print_freq == 0:
            print("\tNeg View Loss: %.4f" % loss.item())
            neg_loss[epoch].append(loss.item())

        #neg_outputs = sim_model(nf)
        #pos_outputs = sim_model(pf)
        #print(neg_outputs.shape)
        #print(pos_outputs.shape)
        #label_1 = torch.ones(args.batch_size).long()
        #label_0 = torch.zeros(args.batch_size).long()
        #label_1 = Variable(label_1.cuda())
        #label_0 = Variable(label_0.cuda())
        # loss = loss + 0.1*sim_loss
        # compute gradient and do SGD step

        #sim_loss = (criterion(pos_outputs, label_0) + criterion(neg_outputs, label_1)) * 0.5
        #sim_loss.backward()
        #print("\tSim Loss: %.4f" % sim_loss.item())
        optimizer.step()
        #sim_optim.step()
        #if (i + 1) % args.print_freq == 0:
        #    print("\tIter [%d/%d] Loss: %.4f" % (i + 1, train_size, loss.item()))


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

            inputs, targets = inputs.cuda(), targets.cuda()
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
    sim_model = nn.DataParallel(sim_model)
    model = nn.DataParallel(model)
    if args.resume:
        load_checkpoint()

    for epoch in range(start_epoch, n_epochs):
        print('\n-----------------------------------')
        print('Epoch: [%d/%d]' % (epoch + 1, n_epochs))
        start = time.time()

        model.train(mode=True)
        scheduler.step()
        scheduler_sim.step()
        train(epoch)
        print('Time taken: %.2f sec.' % (time.time() - start))

        util.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, 'mvcnn')

        # Decaying Learning Rate
        # if (epoch + 1) % args.lr_decay_freq == 0:
        #    lr *= args.lr_decay
        #    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        #    print('Learning rate:', lr)
