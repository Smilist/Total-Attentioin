from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import models.resnet
from utils import progress_bar
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import learning_rate_scheduling as lr_scheduling

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('-m', '--module', type=str, required=True, help='module')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

if args.module == 'se':
    writer = SummaryWriter('/mnt/hdd1/jujeong94/EE898_PA1/trial/log_record/final_se')
elif args.module == 'stn':
    writer = SummaryWriter('/mnt/hdd1/jujeong94/EE898_PA1/trial/log_record/final_stn')
elif args.module == 'joint':
    writer = SummaryWriter('/mnt/hdd1/jujeong94/EE898_PA1/trial/log_record/final_joint')
else:
    writer = SummaryWriter('/mnt/hdd1/jujeong94/EE898_PA1/trial/log_record/final_base')

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'

    if args.module == 'se':
        checkpoint = torch.load('./checkpoint/ckpt_se.t7')
    elif args.module == 'stn':
        checkpoint = torch.load('./checkpoint/ckpt_stn.t7')
    elif args.module == 'joint':
        checkpoint = torch.load('./checkpoint/ckpt_joint.t7')
    else:
        checkpoint = torch.load('./checkpoint/ckpt_base.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('< Module > : ' + args.module)
    print('==> Building model..')
    net = models.resnet.ResNet50(args.module)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

	lr = lr_scheduling.learning_rate(epoch)
	optimizer = optim.SGD(net.parameters(), lr, momentum=0.9, weight_decay=5e-4)

        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    writer.add_scalar('Train_Loss', train_loss/100, epoch)
    writer.add_scalar('Train_Acc', 100.*correct/total, epoch)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

	n = batch_idx
        m = test_loss
        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    writer.add_scalar('Test_Loss', test_loss/100, epoch)
    writer.add_scalar('Test_Acc', 100.*correct/total, epoch)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')

        if args.module == 'se':  
            torch.save(state, './checkpoint/ckpt_se.t7')
        elif args.module == 'stn':
            torch.save(state, './checkpoint/ckpt_stn.t7')
	elif args.module == 'joint':
            torch.save(state, './checkpoint/ckpt_joint.t7')
        else:
            torch.save(state, './checkpoint/ckpt_base.t7')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)

writer.close()
