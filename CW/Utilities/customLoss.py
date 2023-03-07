#PyTorch
#source: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook#Dice-Loss
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, preds, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        preds = torch.sigmoid(preds)       
        
        #flatten label and prediction tensors
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        intersection = (preds * targets).sum()                            
        dice = (2.*intersection + smooth)/(preds.sum() + targets.sum() + smooth)  
        print(1 - dice)
        return 1 - dice
    
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, preds, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        preds = F.sigmoid(preds)       
        
        #flatten label and prediction tensors
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        intersection = (preds * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(preds.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(preds, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE