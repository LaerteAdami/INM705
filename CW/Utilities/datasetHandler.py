import os
import re

import torch
from torch.utils import data
import torch.nn.functional as F

import torchvision.transforms as T
from torchvision import transforms 
from torchvision.utils import make_grid
from torchvision.io.image import read_image
from torchvision.utils import draw_segmentation_masks

from PIL import Image
from PIL import ImageDraw

class CityscapesDataset(data.Dataset):
    '''
    
    Class to handle the dataset. Collects all functionalities to get images, ground truths and labels
    
    '''
    
    def __init__(self, 
                 image_directory = None, 
                 gt_directory = None, 
                 phase = None,
                 trainId = None,
                 colors = None,
                 num_classes = None,
                 resize_dimensions = None):
        
        super(CityscapesDataset, self).__init__()
        
        self.img_dir = image_directory
        self.gt_dir = gt_directory
        self.phase = phase
        self.trainId = trainId
        self.colors = colors
        self.num_classes = num_classes
        self.re_dim = resize_dimensions
        
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
        
        self.t_ = transforms.Compose([transforms.Resize(self.re_dim),
                                      transforms.ToPILImage(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.457, 0.407],
                                                           std=[0.229,0.224,0.225])])
        self.tgt_ = transforms.Compose([transforms.ToPILImage(),
                                      transforms.ToTensor()])
       
    def get_cities(self):
    
        cities = sorted(os.listdir(self.img_dir))
         
        #cities = cities[0:1]
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
        img = read_image(self.img_paths_list[idx]) 
        return self.t_(img)
 
    def get_ground_truth(self, idx):             
        gt_img = read_image(self.mask_paths_list[idx])
        
        gt = torch.zeros_like(gt_img).repeat(self.num_classes, 1, 1)
        #for ind in range(self.num_classes):
        #    gt[ind,:] = ((gt_img==ind)*1) 
            
        for k in self.trainId:
            trainClass = self.trainId.get(k)
            mask_bool = (torch.eq(gt_img, k))*1
            gt[trainClass,:] += mask_bool.squeeze(0)
            
        gt = gt.type(torch.float32)
        tresize_ = transforms.Resize(self.re_dim)
                    
        return tresize_(gt)
    
    
    #### METHOD FOR VISUALIZATION ####     
    def get_image_visualization(self,idx):
        '''
        Method to extract the original image (resized) and its pre-processed version for result visualization
        
        '''
        
        tresize_ = transforms.Resize(self.re_dim)      
        img = read_image(self.img_paths_list[idx]) 
        return tresize_(img), self.t_(img)
      
    ## BASIC METHODS ##
    
    def __len__(self):
        return len(self.img_paths_list)
    
    def __getitem__(self, idx):

        img = self.get_image(idx)
        gt = self.get_ground_truth(idx)

        return idx, img, gt
    
def visualise_and_save(ckp, modelHandler, dataset, image_id, save_name = None):
    '''
    Function to visualise the result of one model and save the segmentation mask
        
    '''
    
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
          
    #x = re.search("_*[0-9]*[.]", ckp)
    #ep = txt[x.start(): x.end()]

    # Load checkpoint
    modelHandler.model.load_state_dict(torch.load(ckp))
    modelHandler.model.eval();

    X_viz, X_viz_norm = dataset.get_image_visualization(image_id)
    gt_viz = dataset.get_ground_truth(image_id)
    gt_viz = gt_viz.argmax(dim=0)

    # Process the input with the model
    pred_viz = modelHandler.model(X_viz_norm.unsqueeze(0).to(device))#['out'] #METTILO PER FCN
    out_viz = F.softmax(pred_viz,dim=1)
    mask = torch.argmax(out_viz, dim=1)

    # Multiple class visualisation
    bool_masks = mask.cpu() == torch.arange(dataset.num_classes)[:, None, None]
    gt_bool_masks = gt_viz.cpu() == torch.arange(dataset.num_classes)[:, None, None]

    sm = draw_segmentation_masks(X_viz, masks = bool_masks, alpha = 0.5,  colors = dataset.colors)
    gt_sm = draw_segmentation_masks(X_viz, masks = gt_bool_masks, alpha = 0.5, colors = dataset.colors)


    sm_list = [sm, gt_sm]

    grid = make_grid(sm_list)
    #to_pil_image(grid)
    
    if save_name is not None:
        
        # Source: https://stackoverflow.com/questions/16373425/add-text-on-image-using-pil
    
        transform = T.ToPILImage()
        img_save = transform(grid)

        ImageDraw.Draw(img_save).text(
            (5, 5),  # Coordinates
            'Prediction',  # Text
            (255, 255, 255)  # Color
        )
        ImageDraw.Draw(img_save).text(
            (261, 5),  # Coordinates
            'Ground truth',  # Text
            (255, 255, 255)  # Color
        )
        img_save.save(save_name)

    return grid
