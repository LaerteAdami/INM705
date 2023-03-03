import numpy as np
import torch

def meanPixelwiseAccuracy(pred, gt):
    '''
    Compute the mean pixelwise accuracy averaged across one batch
    pred dimensions: B x W x H
    gt dimensions: B x C x W x H
    
    PAk = sum( True positives for class k) / (H x W) 
    
    mPA = sum(PAk) / k  --> average on the classes
    
    '''
    
    num_classes = gt.size(1)
    batch_size = pred.size(0)
    
    gt = gt.argmax(dim=1)

    acc = np.zeros((num_classes, batch_size)) # k: channels, B: batches

    for idx, p in enumerate(pred): # loop on the batch
        # p and yy have dimensions: 1 x H x W
        yy = gt[idx]   

        for k in range(num_classes):
            TP = torch.logical_and((p==k), (yy==k)).sum().item() # True positives in the k-th class
            acc[k, idx] = TP / (p.size(0)*p.size(1))

        acc_per_channel = list(np.mean(acc, axis = 1)) # take the mean across batches to have the accuracy per channel

        mPA = np.mean(np.sum(acc, axis = 0)) # take the mean across channels to have the accuracy per batch, then take the mean for the meanPixelwiseAccuracy

    return mPA, acc_per_channel       
    

def meanIoU(pred, gt):
    '''
    Compute the mean Intersection over Union across one batch
    pred dimensions: B x W x H
    gt dimensions: B x C x W x H
    
    IoUk = pred *intersection* gt  / pred *union* gt
    
    mIoU = sum(IoUk) / k  --> average on the classes
    
    '''
    
    num_classes = gt.size(1)
    batch_size = pred.size(0)
    
    gt = gt.argmax(dim=1)

    iou = np.zeros((num_classes, batch_size)) # k: channels, B: batches
 
    for idx, p in enumerate(pred): # loop on the batch
        # p and yy have dimensions: 1 x H x W
        yy = gt[idx]   
        
        for k in range(num_classes):   
            intersection = torch.logical_and((p==k), (yy==k)).sum().item() # pred *intersection* gt in the k-th class
            union = torch.logical_or((p==k), (yy==k)).sum().item() # pred *union* gt in the k-th class
            
            if union != 0:
                iou[k, idx] = intersection / union   
            else:
                iou[k, idx] = 1 # OPPURE 0
                
        iou_per_channel = list(np.mean(iou, axis = 1)) # take the mean across batches to have the IoU per channel

        mIoU = np.mean(np.mean(iou, axis = 0)) # take the mean across channels to have the IoU per batch, then take the mean for the meanIoU

    return mIoU, iou_per_channel       