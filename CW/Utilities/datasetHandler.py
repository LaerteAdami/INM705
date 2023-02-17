from torch.utils import data

class CityscapesDataset(data.Dataset):
    
    def __init__(self, directory, path):
        
        super(CityscapesDataset, self).__init__()
        self.dir = directory
        self.path = path
        
        
        pass
    
    def load_image(self, idx):
        
        pass
    
    
    
    def __len__(self):
        return len(os.listdir(self.dir))