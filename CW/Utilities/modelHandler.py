import torch

class modelFCN:
    
    def __init__(self, model, optimizer, loss_function):
        
        self.model = model
        #self.lr = learning_rate
        self.opt = optimizer
        self.loss_fun = loss_function
        
        
    def train_model(self, dataloader, total_epochs):
        
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')
            
        self.model.to(device)
        
        total_training_loss = []

        self.model.train()

        for e in range(total_epochs):
    
            total_loss_epoch = 0
    
            for idx, batch in enumerate(dataloader):
        
                self.opt.zero_grad()
        
                idx, X, y = batch
                X, y = X.to(device), y.to(device)     
                output = self.model(X)['out']
        
                loss = self.loss_fun(output, y)
                loss.backward()
        
                self.opt.step()
        
                total_loss_epoch += loss
        
            total_loss_epoch/=len(dataloader)
    
            total_training_loss.append(total_loss_epoch.item())
        
            print("Completed epoch {}".format(e))
        
            ## PUT HERE A WAY TO SAVE A CHECKPOINT
            ckp_name = 'city_test_epoch_{}.pth'.format(e)
            torch.save(self.model.state_dict(), ckp_name)
            
        return total_training_loss
            
    def evaluate_model(self):
        
        pass
        
        
        

        
        
        
        
        
        
