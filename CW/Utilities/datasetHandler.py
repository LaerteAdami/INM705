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
                 phase = None,
                 num_classes = None):
        
        super(CityscapesDataset, self).__init__()
        
        self.img_dir = image_directory
        self.gt_dir = gt_directory
        self.phase = phase
        self.num_classes = num_classes
        
        if phase == 'train':       
            self.img_dir = os.path.join(self.img_dir,'train') 
            self.gt_dir = os.path.join(self.gt_dir,'train')
        elif phase == 'val':             
            self.img_dir = os.path.join(self.img_dir,'val')
            self.gt_dir = os.path.join(self.gt_dir,'val')
        elif phase == 'test':             
            self.img_dir = os.path.join(self.img_dir,'test')
            self.gt_dir = os.path.join(self.gt_dir,'test')
         
        self.img_paths_list, self.mask_paths_list = self.get_cities()
        
        #self.imgs = os.listdir(self.img_dir)
        
        self.t_ = transforms.Compose([transforms.Resize((350,500)),
                                      transforms.ToPILImage(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.457, 0.407],
                                                           std=[0.229,0.224,0.225])])
        self.tgt_ = transforms.Compose([#transforms.Resize((350,500)),
                                      transforms.ToPILImage(),
                                      transforms.ToTensor()])
       

    def get_cities(self):
    
        cities = sorted(os.listdir(self.img_dir))
         
        cities = cities[0:1]
        img_paths_list = []
        mask_paths_list = []
        for city in cities:
            city_imgs = sorted(os.listdir(self.img_dir + "/" + city))
            for j in range(len(city_imgs)):
                img_paths_list.append(self.img_dir + '/' + city + '/' + city_imgs[j])
                mask_paths_list.append(self.gt_dir + '/' + city + '/' + city_imgs[j][:len(city) + 15] + 'gtFine_labelIds.png')
        img_paths_list = sorted(list(set(img_paths_list)))
        mask_paths_list = sorted(list(set(mask_paths_list))) 
        
        return img_paths_list, mask_paths_list
    

    
    
    def get_image(self, idx):
        #img = read_image(os.path.join(self.img_dir, self.imgs[idx])) 
        img = read_image(self.img_paths_list[idx]) 
        return self.t_(img)
 
    def get_ground_truth(self, idx):     
        #gt_name = self.imgs[idx][:len(self.imgs[idx])-4] +'.png'       
        #gt_img = self.tgt_(read_image(os.path.join(self.gt_dir,gt_name)))
        
        #gt_img = self.tgt_(read_image(self.mask_paths_list[idx])) # funziona il loop ma non si vedono le classi
        gt_img = read_image(self.mask_paths_list[idx])
        
        gt = torch.zeros_like(gt_img).repeat(self.num_classes, 1, 1)
        for ind in range(self.num_classes):
            gt[ind,:] = ((gt_img==ind)*1) 
            
        gt = gt.type(torch.float32)
        tresize_ = transforms.Resize((350,500))
                    
        return tresize_(gt)
    
    
    #### METHOD FOR VISUALIZATION ####     
    def get_image_visualization(self,idx):
        
        tresize_ = transforms.Resize((350,500))      
        img = read_image(self.img_paths_list[idx]) 
        return tresize_(img), self.t_(img)
        
    
    def get_image_path(self, path):     
        img = read_image(os.path.join(self.img_dir,path))   
        return self.t_(img)
    
    def get_ground_truth_path(self, path):      
        gt = read_image(os.path.join(self.gt_dir,path))       
        return self.tgt_(gt)
    
    
    ## BASIC METHODS ##
    
    def __len__(self):
        return len(self.img_paths_list)
    
    def __getitem__(self, idx):

        img = self.get_image(idx)
        gt = self.get_ground_truth(idx)

        return idx, img, gt