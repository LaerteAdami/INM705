import numpy as np
import torch
import matplotlib.pyplot as plt

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
            denum = (yy==k).sum().item()
            
            if denum == 0:
                if TP == 0:  
                    acc[k, idx] = 1
                else:
                    acc[k, idx] = 0
            else:
                acc[k, idx] = TP / denum

        acc_per_channel = list(np.mean(acc, axis = 1)) # take the mean across batches to have the accuracy per channel

        mPA = np.mean(np.mean(acc, axis = 0)) # take the mean across channels to have the accuracy per batch, then take the mean for the meanPixelwiseAccuracy

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

def metrics_per_class(X, PA, IoU, results_path):  
    
    X_axis = np.arange(len(X))

    plt.bar(X_axis - 0.15, PA, 0.3,label = 'PA')
    plt.bar(X_axis + 0.15, IoU, 0.3,label = 'IoU')

    plt.xticks(X_axis, X)
    plt.xticks(rotation=90)
    #plt.xlabel("Lables")
    plt.ylabel("Metrics")
    plt.legend()
    plt.savefig(results_path+"/metrics_per_class.png")
    plt.show()