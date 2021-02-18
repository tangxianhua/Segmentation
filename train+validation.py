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
# from torch.utils.data import DataLoader
import torch.utils.data as Data
import torchvision.transforms as tr
import torch.nn.parallel
from loss import DICELoss, SoftDiceLoss
from vnet import VNet
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
parser.add_argument('--epochs', type=int, default=1, metavar='N',
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
# %% Loading in the Dataset

with open('/content/gdrive/My Drive/Data/train_data.p3', 'rb') as f:
    x, y = pickle.load(f)
x = x.reshape(x.shape + (1,)).astype(np.float32)
y = y.reshape(y.shape + (1,)).astype(np.float32)  # (50, 256, 256, 32, 1)
print(x.shape)
print(y.shape)
x = torch.Tensor(x)
y = torch.Tensor(y)
x = torch.Tensor.permute(x, (0, 4, 3, 1, 2))
y = torch.Tensor.permute(y, (0, 4, 3, 1, 2))
print(x.shape)
print(y.shape)
# 把数据放在数据库中
torch_dataset = Data.TensorDataset(x, y)
trainloader = Data.DataLoader(
    # 从数据库中每次抽出batch size个样本
    dataset=torch_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=1
)

#validation lodaer
with open('/content/gdrive/My Drive/Data/validation_data.p3', 'rb') as f:
    x, y = pickle.load(f)
x = x.reshape(x.shape + (1,)).astype(np.float32)
y = y.reshape(y.shape + (1,)).astype(np.float32)  # (50, 256, 256, 32, 1)
print(x.shape)
print(y.shape)
x = torch.Tensor(x)
y = torch.Tensor(y)
x = torch.Tensor.permute(x, (0, 4, 3, 1, 2))
y = torch.Tensor.permute(y, (0, 4, 3, 1, 2))
print(x.shape)
print(y.shape)
# 把数据放在数据库中
torch_dataset = Data.TensorDataset(x, y)
validationloder = Data.DataLoader(
    # 从数据库中每次抽出batch size个样本
    dataset=torch_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=1
)



#testloader
with open('', 'rb') as f:
    x, y = pickle.load(f)
x = x.reshape(x.shape + (1,)).astype(np.float32)
y = y.reshape(y.shape + (1,)).astype(np.float32)  # (50, 256, 256, 32, 1)
print(x.shape)
print(y.shape)
x = torch.Tensor(x)
y = torch.Tensor(y)
x = torch.Tensor.permute(x, (0, 4, 3, 1, 2))
y = torch.Tensor.permute(y, (0, 4, 3, 1, 2))
print(x.shape)
print(y.shape)
# 把数据放在数据库中
test_dataset = Data.TensorDataset(x, y)
testloder = Data.DataLoader(
    # 从数据库中每次抽出batch size个样本
    dataset=test_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=1
)


# %% Loading in the model
model = VNet()
if args.cuda:
    model.cuda()
if args.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.5)
if args.optimizer == 'ADAM':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
# Defining Loss Function
criterion = SoftDiceLoss()
testcriterion = SoftDiceLoss()

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

def train(epoch):
    epoch_loss = 0
    #dice = 0
    acc = 0
    sens = 0
    spec = 0
    ppv = 0
    npv = 0
    model.train()
    for batch_idx, (image, mask) in enumerate(trainloader):
        if args.cuda:
            image, mask = image.cuda(), mask.cuda()
        image, mask = Variable(image), Variable(mask)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, mask)
        print('single loss is :', loss.item())
        # print(loss.data)
        loss.backward()
        optimizer.step()

        output = output.data.byte().cpu().numpy()
        mask = mask.data.byte().cpu().numpy()
        TP, FN, FP, TN = confusionmetric(mask, output)
        # print(TN,FP,FN,TP)
        epoch_loss += loss.item()

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

        model_path = os.path.join(r"/content/gdrive/My Drive/Project/myvnet/Results/",
                                  'Trainweight-{}-{}-{}'.format( args.epochs, args.lr))
        torch.save(model.state_dict(), model_path)
        # torch.save(model, model_path)
    epoch_loss = epoch_loss / 50
    Tepoch_loss_list.append(epoch_loss)
    dice = 1 - epoch_loss
    Tdice_list.append(dice)
    acc = acc / 50
    Tacc_list.append(acc)
    sens = sens / 50
    Tsens_list.append(sens)
    spec = spec / 50
    Tspec_list.append(spec)
    ppv = ppv / 50
    Tppv_list.append(ppv)
    npv = npv / 50
    Tnpv_list.append(npv)

    print_format = [epoch, epoch_loss, dice, acc, sens, spec, ppv, npv]
    print(
        '===> Training step {} \tLoss: {:.7f}\tDice: {:.7f}\tAcc: {:.7f}\tSe: {:.7f}\tSp: {:.7f}\tPPV:{:.7f}\tNPV::{:.7f}'.format(
            *print_format))
    with open('/content/gdrive/My Drive/Project/myvnet/Results/Pickle0204-86.txt', 'w') as f:
        f.write(str(Tepoch_loss_list))
        f.write(str(Tdice_list))
        f.write(str(Tacc_list))
        f.write(str(Tsens_list))
        f.write(str(Tspec_list))
        f.write(str(Tppv_list))
        f.write(str(Tnpv_list))
    plt.plot(Tepoch_loss_list)
    plt.savefig("/content/gdrive/My Drive/Project/myvnet/Results/Pickle0204-86.png")

def validation(epoch,bestloss):
    validation_loss = 0
    loader = validationloder

    acc = 0
    sens = 0
    spec = 0
    ppv = 0
    npv = 0

    for batch_idx, (image, mask) in tqdm(enumerate(loader)):
        if args.cuda:
            image, mask = image.cuda(), mask.cuda()
        image,mask = Variable(image),Variable(mask)
        output = model(image)
        validation_loss += testcriterion(output, mask).item()

        output = output.data.byte().cpu().numpy()
        mask = mask.data.byte().cpu().numpy()
        TP, FN, FP, TN = confusionmetric(mask, output)
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
    validation_loss = validation_loss / 50
    Vepoch_loss_list.append(validation_loss)
    dice = 1 - validation_loss
    Vdice_list.append(dice)
    acc = acc / 50
    Vacc_list.append(acc)
    sens = sens / 50
    Vsens_list.append(sens)
    spec = spec / 50
    Vspec_list.append(spec)
    ppv = ppv / 50
    Vppv_list.append(ppv)
    npv = npv / 50
    Vnpv_list.append(npv)
    if validation_loss < Vepoch_loss_list[epoch]:
       bestloss = validation_loss
       print('没有过拟合')
       print('validation loss is ',validation_loss)
       print('best loss is ', bestloss)
       model_path = os.path.join(r"/content/gdrive/My Drive/Project/myvnet/Results/",
                                 'bestweight')
       torch.save(model.state_dict(), model_path)
    else :
      print('过拟合')
      print('best epoch is ',(epoch-1))
      print('best epoch loss is',bestloss)
def test(save_output=True):
    test_loss = 0
    loader = testloder
    path = r'/content/gdrive/My Drive/Project/myvnet/Results/bestweight'
    model.load_state_dict(torch.load(path))
    print('load best weight')
    for batch_idx, (image, mask) in tqdm(enumerate(loader)):
        if args.cuda:
            image, mask = image.cuda(), mask.cuda()
        image,mask = Variable(image),Variable(mask)
        output = model(image)
        test_loss += testcriterion(output, mask).item()
        print('test loss is ',test_loss)
        if save_output:
            np.save(r'/content/gdrive/My Drive/Project/myvnet/Results/{}-Pickle0204.npy'.format(batch_idx),
                    output.data.byte().cpu().numpy())


if args.train:
    #path = r'/content/gdrive/My Drive/Project/myvnet/Results/Pickle0204-1-300-0.0001'
    #model.load_state_dict(torch.load(path))
    #print('load again')
    bestloss = 0
    Tepoch_loss_list = []
    Tacc_list = []
    Tsens_list = []
    Tspec_list = []
    Tppv_list = []
    Tnpv_list = []
    Tdice_list = []

    Vepoch_loss_list = []
    Vacc_list = []
    Vsens_list = []
    Vspec_list = []
    Vppv_list = []
    Vnpv_list = []
    Vdice_list = []

    for epoch in range(args.epochs):
        train(epoch)
        validation(epoch,bestloss)
        # print(loss_list)
    test(save_output=True)

