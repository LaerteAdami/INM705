import numpy as np

def pixelwiseAccuracy(pred, gt):
    
    pred = pred.argmax(dim=1).numpy()
    pred = pred[0,:,:]
    
    gt = gt.numpy()
    gt = gt[0,:,:]
    
    
    
    test = np.zeros_like(gt)

    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            test[i,j] = pred[i,j] == gt[i,j]
            #z = zip(int(ampred[i,j]), int(gt[i,j]))
            
    return sum(sum(test)) / (test.shape[0]*test.shape[1])*100
    
    