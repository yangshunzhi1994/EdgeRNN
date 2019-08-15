'''Train RAVDESS with PyTorch.'''
# 10 crop for data enhancement
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import os
import time
import argparse
import utils
from data.IEMOCAP import IEMOCAP
from data.COMMANDS import COMMANDS
from torch.autograd import Variable
from EdgeRNN import EdgeRNN
from thop import profile
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

parser = argparse.ArgumentParser(description='PyTorch RAVDESS RAVDESS Training')
parser.add_argument('--dataset', type=str, default='IEMOCAP', help='CNN dataset')
parser.add_argument('--train_bs', default=64, type=int, help='learning rate')
parser.add_argument('--test_bs', default=64, type=int, help='learning rate')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
opt = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_PrivateTest_acc = 0  # best PrivateTest accuracy
best_PrivateTest_acc_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

learning_rate_decay_start = 80  # 50
learning_rate_decay_every = 5 # 5
learning_rate_decay_rate = 0.9 # 0.9

total_epoch = 500

total_prediction_fps = 0 
total_prediction_n = 0

# Data
print('==> Preparing data..')
transform_train = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
])

if opt.dataset  == 'COMMANDS':
    print ("This is COMMANDS")
    NUM_CLASSES = 11
    trainset = COMMANDS(split = 'Training', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.train_bs, shuffle=True, num_workers=4)
    PrivateTestset = COMMANDS(split = 'PrivateTest')
    PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=opt.test_bs, shuffle=False, num_workers=1)

else:
    print ("This is IEMOCAP")
    NUM_CLASSES = 4
    trainset = IEMOCAP(split = 'Training', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.train_bs, shuffle=True, num_workers=4)
    PrivateTestset = IEMOCAP(split = 'PrivateTest')
    PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=opt.test_bs, shuffle=False, num_workers=1)
    
net = EdgeRNN(num_class=NUM_CLASSES)    

if opt.dataset  == 'COMMANDS':
    flops, params = profile(net, input_size=(1, 152, 32))
else:
    flops, params = profile(net, input_size=(1, 152, 181))

print("The FLOS of this model is  %0.3f M" % float(flops/1024/1024))
print("The params of this model is  %0.3f M" % float(params/1024/1024))

if opt.dataset == 'IEMOCAP' and os.path.exists('IEMOCAP_Test_model.pt'):
    # Load checkpoint.
    print('==> Resuming from checkpoint IEMOCAP ..')
    
    Private_checkpoint = torch.load('IEMOCAP_Test_model.pt')
    best_PrivateTest_acc = Private_checkpoint['best_PrivateTest_acc']
    best_PrivateTest_acc_epoch = Private_checkpoint['best_PrivateTest_acc_epoch']
    
    print ('best_PrivateTest_acc is '+ str(best_PrivateTest_acc))
    net.load_state_dict(Private_checkpoint['net'], strict=False)
    start_epoch = Private_checkpoint['best_PrivateTest_acc_epoch'] + 1

if opt.dataset == 'COMMANDS' and os.path.exists('COMMANDS_Test_model.pt'):
    # Load checkpoint.
    print('==> Resuming from checkpoint COMMANDS ..')
    
    Private_checkpoint = torch.load('COMMANDS_Test_model.pt')
    best_PrivateTest_acc = Private_checkpoint['best_PrivateTest_acc']
    best_PrivateTest_acc_epoch = Private_checkpoint['best_PrivateTest_acc_epoch']
    
    print ('best_PrivateTest_acc is '+ str(best_PrivateTest_acc))
    net.load_state_dict(Private_checkpoint['net'], strict=False)
    start_epoch = Private_checkpoint['best_PrivateTest_acc_epoch'] + 1
    
else:
    print('==> Building model..')

if use_cuda:
    net.cuda()

criterion = nn.CrossEntropyLoss()
criterion.cuda()
optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    global Train_acc
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))

    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = opt.lr * decay_factor
        utils.set_lr(optimizer, current_lr)  # set the decayed rate
    else:
        current_lr = opt.lr
    print('learning_rate: %s' % str(current_lr))

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()
        train_loss += loss.item()
        
        conf_mat += utils.confusion_matrix(outputs, targets, NUM_CLASSES)
        
        acc = sum([conf_mat[i, i] for i in range(conf_mat.shape[0])])/conf_mat.sum()
        uacc_per_class = [conf_mat[i, i]/conf_mat[i].sum() for i in range(conf_mat.shape[0])]
        unweighted_acc = sum(uacc_per_class)/len(uacc_per_class)

        prec_per_class = [conf_mat[i, i] / conf_mat[:, i].sum() for i in range(conf_mat.shape[0])]
        average_precision = sum(prec_per_class)/len(prec_per_class)

        utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% | unweighted_Acc: %.3f%%'
            % (train_loss/(batch_idx+1), 100.* acc, 100.* unweighted_acc))

    Train_acc = 100.* acc

def PrivateTest(epoch):
    global PrivateTest_acc
    global best_PrivateTest_acc
    global best_PrivateTest_acc_epoch
    global total_prediction_fps
    global total_prediction_n
    net.eval()
    PrivateTest_loss = 0
    correct = 0
    total = 0
    t_prediction = 0
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for batch_idx, (inputs, targets) in enumerate(PrivateTestloader):
        t = time.time()
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        t_prediction += (time.time() - t)
        
        loss = criterion(outputs, targets)
        PrivateTest_loss += loss.item()
        
        conf_mat += utils.confusion_matrix(outputs, targets, NUM_CLASSES)
        
        acc = sum([conf_mat[i, i] for i in range(conf_mat.shape[0])])/conf_mat.sum()
        uacc_per_class = [conf_mat[i, i]/conf_mat[i].sum() for i in range(conf_mat.shape[0])]
        unweighted_acc = sum(uacc_per_class)/len(uacc_per_class)

        prec_per_class = [conf_mat[i, i] / conf_mat[:, i].sum() for i in range(conf_mat.shape[0])]
        average_precision = sum(prec_per_class)/len(prec_per_class)

        utils.progress_bar(batch_idx, len(PrivateTestloader), 'Loss: %.3f | Acc: %.3f%% | unweighted_Acc: %.3f%%'
            % (PrivateTest_loss / (batch_idx + 1), 100. * acc, 100.* unweighted_acc))
    total_prediction_fps = total_prediction_fps + (1 / (t_prediction / len(PrivateTestloader)))
    total_prediction_n = total_prediction_n + 1
    print('Prediction time: %.2f' % t_prediction + ', Average : %.5f/image' % (t_prediction / len(PrivateTestloader)) 
         + ', Speed : %.2fFPS' % (1 / (t_prediction / len(PrivateTestloader))))
    
    # Save checkpoint.
    PrivateTest_acc = 100.* acc
    if PrivateTest_acc > best_PrivateTest_acc:
        print('Saving..')
        print("best_PrivateTest_acc: %0.3f" % PrivateTest_acc)
        state = {
            'net': net.state_dict() if use_cuda else net,
            'best_PrivateTest_acc': PrivateTest_acc,
            'best_PrivateTest_acc_epoch': epoch,
        }
        if opt.dataset == 'COMMANDS':
            torch.save(state, 'COMMANDS_Test_model.pt')
        else:
            torch.save(state, 'IEMOCAP_Test_model.pt')
        best_PrivateTest_acc = PrivateTest_acc
        best_PrivateTest_acc_epoch = epoch

for epoch in range(start_epoch, total_epoch):
    train(epoch)
    PrivateTest(epoch)

print("best_PrivateTest_acc: %0.3f" % best_PrivateTest_acc)
print("best_PrivateTest_acc_epoch: %d" % best_PrivateTest_acc_epoch)

print("total_prediction_fps: %0.2f" % total_prediction_fps)
print("total_prediction_n: %d" % total_prediction_n)
print('Average speed: %.2f FPS' % (total_prediction_fps / total_prediction_n))



    