import numpy as np

def meanPixelwiseAccuracy(pred, gt):
    '''
    Compute the mean pixelwise accuracy averaged across one batch
    pred and gt dimensions: B x C x W x H
    
    PAk = sum( True positives for class k) / (H x W) 
    
    mPA = sum(PAk) / k  --> average on the classes
    
    '''

    acc = [] # every entry is the mPA for one image in the batch
    
    for idx, p in enumerate(pred):      
        # p and yy have dimensions: C x H x W
        yy = gt[idx]   
        
        TPk = ((p == yy)*1).sum() # true positive across all dimensions C, H and W
    
        acc.append(  TPk / (p.size(0)*p.size(1)*p.size(2)) ) # divided by all pixels and averaged over the classes
            
    return (sum(acc) / len(acc)).item() # take the average across batch size
    

    
def meanIoU(pred, gt):
    '''
    Compute the mean Intersection over Union across one batch
    pred and gt dimensions: B x C x W x H
    
    IoUk = pred *intersection* gt  / pred *union* gt
    
    mIoU = sum(IoUk) / k  --> average on the classes
    
    '''
    
    channels = pred.size(1)
    batches = pred.size(0)
    
    iou = []
    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy() 
 
    for idx in range(batches):      
        y = gt[idx] 
        p = pred[idx]
        iouk = []

        for channel in range(channels):
            intersection = np.logical_and(p[channel,:], y[channel,:])
            union = np.logical_or(p[channel,:], y[channel,:])
            if np.sum(union)!=0:
                iouk.append( np.sum(intersection) / (np.sum(union)))

        iou.append(sum(iouk)/ len(iouk))
   
    return (sum(iou) / len(iou)).item()