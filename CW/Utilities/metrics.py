import numpy as np

def pixelwiseAccuracy(pred, gt):
    
    acc = []
    
    for idx, p in enumerate(pred): 
        p = p#.argmax(dim=0)
        yy = gt[idx]#.argmax(dim=0)   
    
        acc.append( ((p == yy)*1).sum() / (p.size(0)*p.size(1)*p.size(2)) )
            
    return (sum(acc) / len(acc)).item() # take the average across batch size
    
    