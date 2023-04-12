import torch
from Utilities.metrics import *
from torch.nn.functional import softmax

class modelFCN:
    
    def __init__(self, model, optimizer, loss_function):
        
        self.model = model
        self.opt = optimizer
        self.loss_fun = loss_function
        
        
    def train_model(self, dataloader, total_epochs, save_every_epochs, ckp_name):
        
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')
            
        self.model.to(device)
        
        total_training_loss = []
        ckp_list = []

        self.model.train()

        for e in range(1,total_epochs+1):
    
            total_loss_epoch = 0
    
            for id_batch, batch in enumerate(dataloader):
        
                self.opt.zero_grad()
        
                idx, X, y = batch
                X, y = X.to(device), y.to(device)     
                output = self.model(X)
               
                loss = self.loss_fun(output, y)
                loss.backward()
        
                self.opt.step()
        
                total_loss_epoch += loss
        
            total_loss_epoch/=len(dataloader)
            
            total_training_loss.append(total_loss_epoch.item())
        
            print("Completed epoch {}".format(e))
        
            ## SAVE A CHECKPOINT
            if e%save_every_epochs == 0: # save the model every "save_every_epochs" epochs
                ckp_path = ckp_name+'_{}.pth'.format(e)
                torch.save(self.model.state_dict(), ckp_path)
                ckp_list.append(ckp_path)
            
        return total_training_loss, ckp_list
        
    def evaluate_model(self, valloader, ckp):
        
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')

        # Load checkpoint
        self.model.load_state_dict(torch.load(ckp))
        self.model.eval()
        self.model.to(device)

        _, X_test, y_test = valloader
        X_test, y_test = X_test.to(device), y_test.to(device) 

        pred = self.model(X_test)

        # Apply softmax and threshold value
        if not torch.is_tensor(pred):
            pred = pred['out']
            
        out = softmax(pred,dim=1)# > 0.5
        out = torch.argmax(out, dim=1)

        # Computing metrics
        mPA, PAk = meanPixelwiseAccuracy(out, y_test)
        mIoU, IoUk = meanIoU(out, y_test)

        return mPA, PAk, mIoU, IoUk    
    

    