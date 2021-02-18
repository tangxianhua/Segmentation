# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.functional as f
import numpy as np


class DICELossMultiClass(nn.Module):

    def __init__(self):
        super(DICELossMultiClass, self).__init__()

    def forward(self, output, mask):
        num_classes = output.size(1)
        dice_eso = 0
        for i in range(num_classes):
            probs = torch.squeeze(output[:, i, :, :], 1)
            mask = torch.squeeze(mask[:, i, :, :], 1)

            num = probs * mask
            num = torch.sum(num, 2)
            num = torch.sum(num, 1)

            # print( num )

            den1 = probs * probs
            # print(den1.size())
            den1 = torch.sum(den1, 2)
            den1 = torch.sum(den1, 1)

            # print(den1.size())

            den2 = mask * mask
            # print(den2.size())
            den2 = torch.sum(den2, 2)
            den2 = torch.sum(den2, 1)

            # print(den2.size())
            eps = 0.0000001
            dice = 2 * ((num + eps) / (den1 + den2 + eps))
            # dice_eso = dice[:, 1:]
            dice_eso += dice

        loss = 1 - torch.sum(dice_eso) / dice_eso.size(0)
        return loss


class DICELoss(nn.Module):

    def __init__(self):
        super(DICELoss, self).__init__()

    def forward(self, output, mask):

        probs = torch.squeeze(output, 1)
        mask = torch.squeeze(mask, 1)

        intersection = probs * mask
        intersection = torch.sum(intersection, 2)
        intersection = torch.sum(intersection, 1)

        den1 = probs * probs
        den1 = torch.sum(den1, 2)
        den1 = torch.sum(den1, 1)

        den2 = mask * mask
        den2 = torch.sum(den2, 2)
        den2 = torch.sum(den2, 1)

        eps = 1e-8
        dice = 2 * ((intersection + eps) / (den1 + den2 + eps))
        # dice_eso = dice[:, 1:]
        dice_eso = dice

        loss = 1 - torch.sum(dice_eso) / dice_eso.size(0)
        return loss


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, output, mask):
        num = mask.size(0)
        smooth = 1e-8

        # probs = f.sigmoid(logits)
        m1 = output.view(num, -1)
        m2 = mask.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score
class Myloss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Myloss,self).__init__()

    def forward(self,output,mask):
        num = mask.size(0)
        smooth = 1e-8

        # probs = f.sigmoid(logits)
        m1 = output.view(num, -1)
        m2 = mask.view(num, -1)
        intersection = (m1 * m2)

        score = (intersection.sum(1) + smooth) / (m2.sum(1) + smooth)

        return score