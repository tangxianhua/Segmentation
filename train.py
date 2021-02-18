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
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# import torch.utils.data as Data
from torch.utils.data import DataLoader
import torchvision.transforms as tr
import torch.nn.parallel
from trainloss import SoftDiceLoss
from traindata import MergeTrain, MergeVali
# from trainmodel import VNet,MaskVNet
from modelmerge import MaskVNetmerge
from tqdm import tqdm
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix

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
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
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
parser.add_argument('--optimizer', type=str, default='ADAM', metavar='str',
                    help='Optimizer (default: SGD)')
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()
DATA_FOLDER = args.data_folder

weightsavepath = '/content/gdrive/My Drive/Result2/mergenewdatanoSF'
traintxtsavepath = '/content/gdrive/My Drive/Result2/mergenewdatanoSF'
traintxtname = '1trainmergenewdatanoSF150.txt'
trainpngsavepath = '/content/gdrive/My Drive/Result2/mergenewdatanoSF'

trainpngname = '1trainmergenewdatanoSF150.png'
traincenterpngsavepath = '/content/gdrive/My Drive/Result2/mergenewdatanoSF'
traincenterpngname = '1trainmergenewdatanoSF150.png'
trainallpngsavepath = '/content/gdrive/My Drive/Result2/mergenewdatanoSF'
trainallpngname = '1trainmergenewdatanoSF150.png'

# ********************************训练数据*********************************************
print('********************************训练数据*********************************************')
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@导入训练数据1@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
dset_train = Mytraincenter(DATA_FOLDER, train=True, transform=tr.ToTensor())
trainloader = DataLoader(dset_train, batch_size=args.batch_size, shuffle=False, num_workers=1)
print("Training Data 1 : ", len(trainloader.dataset))

# ********************************验证数据*********************************************
print('********************************验证数据*********************************************')
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@验证数据1@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
dset_vali = Myvali(DATA_FOLDER, transform=tr.ToTensor())
valiloader = DataLoader(dset_vali, batch_size=args.batch_size, shuffle=False, num_workers=1)
print("Validation Data 1 : ", len(valiloader.dataset))

# ********************************测试数据*********************************************
# dset_test = Mytrain(DATA_FOLDER, train=6, transform=tr.ToTensor())
# test_loader = DataLoader(dset_test, batch_size=args.test_batch_size, shuffle=False, num_workers=1)
# print("Test Data :", len(test_loader.dataset))

# ******************************************************************************************
model = MaskVNetmerge()
# centermodel = MaskVNet()
if args.cuda:
    model.cuda()
    # centermodel.cuda()
if args.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.5)
if args.optimizer == 'ADAM':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
# Defining Loss Function
criterion = SoftDiceLoss()
testcriterion = SoftDiceLoss()
crosscriterion = nn.BCELoss()


def confusionmetric(X, Y):
    x1 = X.reshape(-1)  # pickle mask最大值为1，mhd为255
    # print('x1 is:',np.max(x1))
    y1 = Y.reshape(-1)
    # print('y1 is:', np.max(y1))
    conmat = confusion_matrix(x1, y1)
    com = conmat.flatten()
    TN = com[0]
    FP = com[1]
    FN = com[2]
    TP = com[3]
    return TN, FP, FN, TP


# **************************************训练****************************************
# *******************************************************************************************
# ****************************** train1 ***************************************************

def train(epoch):
    print('******************************train epoch******************', epoch)
    trainloss = 0
    allloss = 0
    centerloss = 0
    acc = 0
    sens = 0
    spec = 0
    ppv = 0
    npv = 0

    model.train()
    for batch_idx, (image, mask, center) in enumerate(trainloader):
        if args.cuda:
            image, mask, center = image.cuda(), mask.cuda(), center.cuda()
        image, mask, center = Variable(image), Variable(mask), Variable(center)
        optimizer.zero_grad()
        output1, output2 = model(image)
        loss1 = criterion(output1, mask)
        print('Train loss is :', loss1.item())
        losscenter = crosscriterion(output2, center)
        print('center single loss is :', losscenter.item())

        loss = loss1 + losscenter
        loss.backward()
        optimizer.step()
        output1 = output1.data.byte().cpu().numpy()
        mask = mask.data.byte().cpu().numpy()
        TP, FN, FP, TN = confusionmetric(mask, output1)

        trainloss += loss1.item()
        allloss += loss.item()
        centerloss += losscenter.item()
        ACC = (TP + TN) / (TP + FN + FP + TN)
        acc += ACC
        SENS = TP / (TP + FN)
        sens += SENS
        SPEC = TN / (TN + FP)
        spec += SPEC
        PPV = TP / (TP + FP + 1e-10)
        ppv += PPV
        NPV = TN / (TN + FN + 1e-10)
        npv += NPV

        model_path = os.path.join(weightsavepath, '1trainmergenewdatanoSF150--{}-{}'.format(args.epochs, args.lr))
        torch.save(model.state_dict(), model_path)
    trainloss = trainloss / 40
    allloss = allloss / 40
    centerloss = centerloss / 40
    Tdice = 1 - trainloss
    print('Train mean loss is', trainloss)
    acc = acc / 40
    sens = sens / 40
    spec = spec / 40
    ppv = ppv / 40
    npv = npv / 40

    Tloss_list.append(trainloss)
    Tallloss_list.append(allloss)
    Tcenterloss_list.append(centerloss)
    Tdice_list.append(Tdice)
    Tacc_list.append(acc)
    Tsens_list.append(sens)
    Tspec_list.append(spec)
    Tppv_list.append(ppv)
    Tnpv_list.append(npv)

    print_format = [epoch, trainloss, centerloss, allloss, Tdice, acc, sens, spec, ppv, npv]
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print(
        '===> Training step {} \tLoss: {:.7f} \tCenterloss::{:.7f} \tAllloss::{:.7f} \tDice: {:.7f}\tAcc: {:.7f}\tSe: {:.7f}\tSp: {:.7f}\tPPV:{:.7f}\tNPV::{:.7f}'.format(
            *print_format))
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    with open(os.path.join(traintxtsavepath, str(traintxtname)), 'w') as f:
        f.write(str(Tloss_list))
        f.write(str(Tallloss_list))
        f.write(str(Tcenterloss_list))
        f.write(str(Tdice_list))
        f.write(str(Tacc_list))
        f.write(str(Tsens_list))
        f.write(str(Tspec_list))
        f.write(str(Tppv_list))
        f.write(str(Tnpv_list))
    plt.plot(Tloss_list)
    plt.savefig(os.path.join(trainpngsavepath, str(trainpngname), ))
    plt.plot(Tallloss_list)
    plt.savefig(os.path.join(trainallpngsavepath, str(trainallpngname)))
    plt.plot(Tcenterloss_list)
    plt.savefig(os.path.join(traincenterpngsavepath, str(traincenterpngname)))


def vali():
    print('******************************validation5*******************************')
    valiloss = 0
    acc = 0
    sens = 0
    spec = 0
    ppv = 0
    npv = 0
    vloader = valiloader
    for batch_idx, (image, mask) in tqdm(enumerate(vloader)):
        if args.cuda:
            image, mask = image.cuda(), mask.cuda()
        image, mask = Variable(image), Variable(mask)
        output1, output2 = model(image)
        valiloss += testcriterion(output1, mask).item()

        output1 = output1.data.byte().cpu().numpy()
        mask = mask.data.byte().cpu().numpy()
        TP, FN, FP, TN = confusionmetric(mask, output1)
        # print(TN,FP,FN,TP)
        ACC = (TP + TN) / (TP + FN + FP + TN)
        acc += ACC
        SENS = TP / (TP + FN)
        sens += SENS
        SPEC = TN / (TN + FP)
        spec += SPEC
        PPV = TP / (TP + FP + 1e-10)
        ppv += PPV
        NPV = TN / (TN + FN + 1e-10)
        npv += NPV
    valiloss = valiloss / 10
    Vdice = 1 - valiloss
    acc = acc / 10
    sens = sens / 10
    spec = spec / 10
    ppv = ppv / 10
    npv = npv / 10

    Vloss_list.append(valiloss)
    Vdice_list.append(Vdice)
    Vacc_list.append(acc)
    Vsens_list.append(sens)
    Vspec_list.append(spec)
    Vppv_list.append(ppv)
    Vnpv_list.append(npv)

    print_format = [valiloss, Vdice, acc, sens, spec, ppv, npv]
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print(
        '===> Training step  \tLoss: {:.7f}\tDice: {:.7f}\tAcc: {:.7f}\tSe: {:.7f}\tSp: {:.7f}\tPPV:{:.7f}\tNPV::{:.7f}'.format(
            *print_format))
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    with open(os.path.join('/content/gdrive/My Drive/Result2/mergenewdatanoSF', '1valimergenewdatanoSF150.txt'),
              'w') as f:
        f.write(str(Vloss_list))
        f.write(str(Vdice_list))
        f.write(str(Vacc_list))
        f.write(str(Vsens_list))
        f.write(str(Vspec_list))
        f.write(str(Vppv_list))
        f.write(str(Vnpv_list))
    plt.plot(Vloss_list)
    plt.savefig(os.path.join('/content/gdrive/My Drive/Result2/mergenewdatanoSF', '1valimergenewdatanoSF150.png'))


# **********************************************测试*********************************************8
# def test(save_output=True):
#     test_loss = 0
#     loader = testloder
#     # path = r'/content/drive/My Drive/Project/myvnet/Results/cebestweight'
#     # print('load best weight')
#     # model.load_state_dict(torch.load(path))

#     for batch_idx, (image, mask) in tqdm(enumerate(loader)):
#         if args.cuda:
#             image, mask = image.cuda(), mask.cuda()
#         image, mask = Variable(image), Variable(mask)
#         output = model(image)
#         test_loss += testcriterion(output, mask).item()
#         print('test loss is ', test_loss)
#         if save_output:
#             np.save(r'/content/gdrive/My Drive/Project/myvnet/Results/KfoldResults/{}-cekfold.npy'.format(batch_idx),
#                     output.data.byte().cpu().numpy())


# **********************************************************************************************************
if args.train:

    Tloss_list = []
    Tallloss_list = []
    Tcenterloss_list = []
    Tdice_list = []
    Tacc_list = []
    Tsens_list = []
    Tspec_list = []
    Tppv_list = []
    Tnpv_list = []

    Vloss_list = []
    Vdice_list = []
    Vacc_list = []
    Vsens_list = []
    Vspec_list = []
    Vppv_list = []
    Vnpv_list = []
    # path = r'/content/gdrive/My Drive/Result2/mergenewdatanoSF/1trainmergenewdatanoSF140--10-0.0001'
    # model.load_state_dict(torch.load(path))
    # print('load again')

    for epoch in range(args.epochs):
        train(epoch)
    vali()
    # test(save_output=True)

