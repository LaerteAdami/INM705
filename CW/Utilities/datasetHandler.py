import os
import torch
from torch.utils import data
from PIL import Image
from torchvision import transforms 
from torchvision.io.image import read_image
from bs4 import BeautifulSoup 

class CityscapesDataset(data.Dataset):
    '''
    
    Class to handle the dataset. Collects all functionalities to get images, ground truths and labels
    
    '''
    
    def __init__(self, 
                 image_directory = None, 
                 gt_directory = None, 
                 label_directory = None,
                 phase = None,
                 num_classes = None):
        
        super(CityscapesDataset, self).__init__()
        
        self.img_dir = image_directory
        self.gt_dir = gt_directory
        self.label_dir = label_directory
        self.phase = phase
        self.num_classes = num_classes
        
        if phase == 'train':       
            self.img_dir = os.path.join(self.img_dir,'test_data')   
        elif phase == 'val':             
            self.img_dir = os.path.join(self.img_dir,'val_data')
        elif phase == 'test':             
            self.img_dir = os.path.join(self.img_dir,'test_data')
             
        self.imgs = os.listdir(self.img_dir)
        self.t_ = transforms.Compose([transforms.Resize((350,500)),
                                      transforms.ToPILImage(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.457, 0.407],
                                                           std=[0.229,0.224,0.225])])
        self.tgt_ = transforms.Compose([transforms.Resize((350,500)),
                                      transforms.ToPILImage(),
                                      transforms.ToTensor()])
       
    def get_image(self, idx):
        img = read_image(os.path.join(self.img_dir, self.imgs[idx]))       
        return self.t_(img)
 
    def get_ground_truth(self, idx):     
        gt_name = self.imgs[idx][:len(self.imgs[idx])-4] +'.png'       
        gt_img = self.tgt_(read_image(os.path.join(self.gt_dir,gt_name)))
        gt = torch.zeros_like(gt_img).repeat(self.num_classes, 1, 1)
        for ind in range(self.num_classes):
            gt[ind,:] = ((gt_img==ind)*1) 
        return gt       
         
        
        
    def get_image_path(self, path):     
        img = read_image(os.path.join(self.img_dir,path))   
        return self.t_(img)
    
    def get_ground_truth_path(self, path):      
        gt = read_image(os.path.join(self.gt_dir,path))       
        return self.tgt_(gt)
    
    
    ## BASIC METHODS ##
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):

        img = self.get_image(idx)
        gt = self.get_ground_truth(idx)

        return idx, img, gt