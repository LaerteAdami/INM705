import torch
import torch.nn as nn
from torch.nn import functional as F

class modelConstructor(nn.Module):
    
    def __init__(self, backbone, decoder, resize_dims = None):
        
        super(modelConstructor, self).__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.resize_dims = resize_dims
    
    def forward(self, x):
        
        x = self.backbone(x)
        
        if torch.is_tensor(x):
            x = self.decoder(x)
        else:
            x = self.decoder(x["out"]) # OrderDict from Resnet50 backbone
            
        if self.resize_dims is not None:
            out = F.interpolate(x, size=self.resize_dims, mode="bilinear", align_corners=False)
        else:
            out = x
            
        return out