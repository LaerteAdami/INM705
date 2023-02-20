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
        
            ## PUT HERE A WAY TO SAVE A CHECKPOINT
        return total_training_loss
            
    def evaluate_model(self):
        
        pass
        
        
        

        
        
        
        
        
        
        for e in range(total_epochs):
            total_loss_epoch = 0
            for id, batch in enumerate(dataloader):
                optimizer.zero_grad()
                # this should be of dimensionalities (20, BxCxHxW, Bx20)
                idx, X, y = batch

                X,y  = X.to(device), y.to(device)
                output = model(X)
                #print(output>0, y)
                loss = loss_function(output,y)
                loss.backward()
                optimizer.step()
            total_loss_epoch += loss