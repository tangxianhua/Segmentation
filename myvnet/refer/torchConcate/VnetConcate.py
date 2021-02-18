# %% -*- coding: utf-8 -*-
'''
Author: Shreyas Padhy
Driver file for Standard UNet Implementation
'''
from __future__ import print_function
import argparse
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as tr

import torch.nn.parallel
from VnetData import DataVnetConcate
from VnetData import DataVnet
from vnetlosses import DICELoss

from VnetConcateModel import VNet
from VnetConcateModel import VNet3
from tqdm import tqdm
import numpy as np
import skimage.io as io
from sklearn .metrics import confusion_matrix
# %% import transforms

# %% Training settings
parser = argparse.ArgumentParser(
    description='UNet + BDCLSTM for BraTS Dataset')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--train', action='store_true', default=True,
                    help='Argument to train model (default: False)')
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training (default: False)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='batches to wait before logging training status')
parser.add_argument('--size', type=int, default=256, metavar='N',
                    help='imsize')
parser.add_argument('--load', type=str, default=None, metavar='str',
                    help='weight file to load (default: None)')
parser.add_argument('--data-folder', type=str, default='./Data/', metavar='str',
                    help='folder that contains data (default: test dataset)')
parser.add_argument('--save', type=str, default='OutMasks', metavar='str',
                    help='Identifier to save npy arrays with')
parser.add_argument('--optimizer', type=str, default='SGD', metavar='str',
                    help='Optimizer (default: SGD)')
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()
DATA_FOLDER = args.data_folder
# %% Loading in the Dataset

dset_train = DataVnetConcate(DATA_FOLDER, train=True, transform=tr.ToTensor())
train_loader = DataLoader(dset_train, batch_size=args.batch_size, shuffle=False, num_workers=1)
dset_test = DataVnetConcate(DATA_FOLDER, train=False, transform=tr.ToTensor())
test_loader = DataLoader(dset_test, batch_size=args.test_batch_size, shuffle=False, num_workers=1)

print("Training Data : ", len(train_loader.dataset))
print("Test Data :", len(test_loader.dataset))

# %% Loading in the model
model = VNet3()
if args.cuda:
    model.cuda()
if args.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.5)
if args.optimizer == 'ADAM':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
# Defining Loss Function
criterion = DICELoss()
# criterion = nn.CrossEntropyLoss()

def confusionmetric(X,Y):
    x1 = X.reshape(-1)//255
    # print('x1 is:',np.max(x1))
    y1 = Y.reshape(-1)
    # print('y1 is:', np.max(y1))
    conmat = confusion_matrix(x1,y1)
    com = conmat.flatten()
    TN = com[0]
    FP = com[1]
    FN = com[2]
    TP = com[3]

    return  TN,FP,FN,TP
epoch_loss_list =[]
epoch_lossx_list =[]
epoch_lossy_list =[]
epoch_lossz_list =[]
acc_list = []
sens_list = []
spec_list = []
ppv_list = []
npv_list = []
dice_list = []
model.train()
def train(epoch, loss_list):
    epoch_loss = 0
    epoch_lossx = 0
    epoch_lossy = 0
    epoch_lossz = 0
    dice = 0
    acc =0
    sens =0
    spec= 0
    ppv=0
    npv=0

    for batch_idx, (imagex, imagey,imagez,mask) in enumerate(train_loader):
        if args.cuda:
            imagex,imagey,imagez, mask = imagex.cuda(), imagey.cuda(),imagez.cuda(),mask.cuda()
        imagex,imagey,imagez, mask = Variable(imagex),Variable(imagey),Variable(imagez), Variable(mask)
        optimizer.zero_grad()#optimzier使用之前需要zero清零一下，因为如果不清零，那么使用的这个grad就得同上一个mini-batch有关
        outputx,outputy,outputz = model(imagex,imagey,imagez)

        lossx = criterion(outputx, mask)
        lossy = criterion(outputy, mask)
        lossz = criterion(outputz, mask)
        print('single loss is :',lossx.item(),lossy.item(),lossz.item())
        #print(loss.data)

        loss = lossx+lossy+lossz
        loss.backward()
        # lossx.backward()
        # lossy.backward()
        # lossz.backward()
        optimizer.step()

        outputx = outputx.data.byte().cpu().numpy()
        mask = mask.data.byte().cpu().numpy()
        TP, FN, FP, TN= confusionmetric(mask,outputx)

        # print(TN,FP,FN,TP)
        epoch_loss += loss
        epoch_lossx += lossx
        epoch_lossy += lossy
        epoch_lossz += lossz
        #print('epoch_loss is:',epoch_loss)
        ACC = (TP+TN)/(TP+FN+FP+TN)
        acc +=ACC
        SENS = TP/(TP+FN)
        sens +=SENS
        SPEC = TN/(TN+FP)
        spec +=SPEC
        PPV = TP/(TP+FP+1e-10)
        ppv += PPV
        NPV = TN/(TN+FN+1e-10)
        npv += NPV

        # epoch_loss += loss.data#tensor
        model_path = os.path.join(r"/home/fafafa/pytorch3d/VNET/weight",
                                  'VC12201322-STATE-{}-{}-{}'.format(args.batch_size, args.epochs, args.lr))
        torch.save(model.state_dict(), model_path)
        #torch.save(model, model_path)
    epoch_loss=epoch_loss/50
    epoch_loss_list.append(epoch_loss)
    epoch_lossx = epoch_lossx / 50
    epoch_lossx_list.append(epoch_lossx)
    epoch_lossy = epoch_lossy / 50
    epoch_lossy_list.append(epoch_lossy)
    epoch_lossz = epoch_lossz / 50
    epoch_lossz_list.append(epoch_lossz)
    dice  = 1-epoch_loss
    dice_list.append(dice)
    acc =acc/50
    acc_list.append(acc)
    sens =sens/50
    sens_list.append(sens)
    spec=spec/50
    spec_list.append(spec)
    ppv= ppv/50
    ppv_list.append(ppv)
    npv=npv/ 50
    npv_list.append(npv)
    model.train()
    print_format = [epoch, epoch_loss, dice, acc, sens, spec,ppv,npv ]
    print(
        '===> Training step {} \tLoss: {:.7f}\tDice: {:.7f}\tAcc: {:.7f}\tSe: {:.7f}\tSp: {:.7f}\tPPV:{:.7f}\tNPV::{:.7f}'.format(*print_format))
    with open('VC12201322-STATE.txt', 'w') as f:
        f.write(str(epoch_loss_list))
        f.write(str(dice_list))
        f.write(str(acc_list))
        f.write(str(sens_list))
        f.write(str(spec_list))
        f.write(str(ppv_list))
        f.write(str(npv_list))
    plt.plot(epoch_loss_list)
    plt.plot(epoch_lossx_list)
    plt.plot(epoch_lossy_list)
    plt.plot(epoch_lossz_list)
    plt.savefig("VC12201322-STATE.png")
    #     # Compute accuracy
    #     accuracy = output.eq(mask).cpu().sum() / mask.numel()#accuracy的计算把判断正确的个数累加起来
    #     epoch_acc += accuracy
    # epoch_loss = epoch_loss/ 50
    # epoch_acc = epoch_acc/ 50
    # print_format = [epoch, epoch_loss]
    # print('===> Training step {} \tLoss: {:.5f}\tAccuracy: {:.5f}'.format(*print_format))
# lossy = hist.history['loss']
#     plt.plot(lossy)
#     plt.savefig("1120-ADD-F-500-loss-dice--.png")


def test(train_accuracy=False, save_output=False):
    test_loss = 0
    if train_accuracy:
        loader = train_loader
    else:
        loader = test_loader
    for batch_idx, (imagex,imagey,imagez, mask) in tqdm(enumerate(loader)):
        if args.cuda:
        #     image,imagey,imagez, mask = image.cuda(),imagey.cuda(),imagez.cuda(), mask.cuda()
        # image,imagey,imagez, mask = Variable(image),Variable(imagey),Variable(imagez), Variable(mask)
        # output = model(image,imagey,imagez)
            imagex,imagey,imagez, mask = imagex.cuda(),imagey.cuda(),imagez.cuda(), mask.cuda()
        imagex,imagey,imagez, mask = Variable(imagex),Variable(imagey),Variable(imagez), Variable(mask)
        outputx,outputy,outputz = model(imagex,imagey,imagez)
        # test_loss += criterion(output, mask).data[0]
        # maxes, out = torch.max(output, 1, keepdim=True)
        if save_output and (not train_accuracy):
            np.save(r'/home/fafafa/pytorch3d/VNET/np/{}-VC12201322-STATE-500.npy'.format(batch_idx), outputx.data.byte().cpu().numpy())
        # test_loss += criterion(output, mask).item()
    # Average Dice Coefficient
    # test_loss /= len(loader)
    # if train_accuracy:
    #     print('\nTraining Set: Average DICE Coefficient: {:.4f})\n'.format(test_loss))
    # else:
    #     print('\nTest Set: Average DICE Coefficient: {:.4f})\n'.format(test_loss))

if args.train:
    loss_list = []
    path = r'/home/fafafa/pytorch3d/VNET/weight/VC12201322-STATE-1-500-0.1'
    model.load_state_dict(torch.load(path))
    print('load again')
    for i in tqdm(range(args.epochs)):
        train(i, loss_list)
        #print(loss_list)
    test(save_output=True)

