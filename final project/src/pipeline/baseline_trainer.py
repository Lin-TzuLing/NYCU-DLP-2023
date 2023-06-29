import os
import torch
import torch.nn as nn
from tqdm import tqdm



class baseline_trainer():
    def __init__(self, args, framework, device):
        super(baseline_trainer).__init__()

        self.args = args
        self.lr = args.joint_lr
        self.epochs = args.joint_epochs
        self.device = device
        self.framework = framework
        self.baseline_name = os.path.join(args.ckpt_path,"{}_{}_baseline.pth".format(self.args.arch, self.args.pretrained))
        if self.args.mode=='baseline_test':
            self.load_framework(self.baseline_name)

        # Loss function (only classfication task in baseline)
        self.criterion_class = nn.CrossEntropyLoss()
    
        # Optimizer & Scheduler
        self.params = (list(self.framework["model"].parameters())+
                       list(self.framework["downstream_classifier"].parameters()))
        
        # self.optimizer = torch.optim.SGD(self.params, lr=self.lr, momentum=0.9, weight_decay=5e-4)
        self.optimizer = torch.optim.Adam(self.params, lr=self.lr, weight_decay=5e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                                       factor=0.8, patience=5, 
                                                                       min_lr=1e-7,  verbose=True)
        

    def train(self, train_loader, test_loader):        
        best_acc_epoch = 0 
        best_test_acc = 0.0
        y_pred, y_true = [], []
        f = open(f"{self.args.record_path}/{self.args.arch}_{self.args.pretrained}_baseline.txt", "w")
        
        # training epochs
        for epoch in tqdm(range(self.epochs)):
            # training all network
            self.framework["model"].train()
            self.framework["downstream_classifier"].train()

            total_loss_class, correct = 0.0, 0.0

            # training batch
            for _, (inputs, label, _, _, _) in enumerate(tqdm(train_loader, leave=False)):
                inputs = inputs.to(self.device, dtype=torch.float)
                label = label.to(self.device, dtype=torch.long)
                               
                feature = self.framework["model"](inputs)
                pred_class = self.framework["downstream_classifier"](feature)

                if (label.shape)[0]==1:
                    pred_class = pred_class.unsqueeze(0)

                loss_class = self.criterion_class(pred_class, label)
                total_loss_class += loss_class.item()

                # train classfication task independently
                loss = loss_class

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pred = pred_class.argmax(dim=1)
                for i in range(len(label)):
                    y_pred.append(pred[i].item())
                    y_true.append(label[i].item())
                    if pred[i] == label[i]:
                        correct +=1

            # per batch
            train_acc = 100.*correct/len(train_loader.dataset)

            # evaluation
            test_acc_class = self.test(test_loader)
            
            # save model
            if test_acc_class > best_test_acc:
                best_test_acc = test_acc_class
                best_acc_epoch = epoch
                self.save_framework(self.baseline_name)
                tqdm.write("(baseline) Best Classification Model saved!")  
            
            # write records
            f.write("epoch:{}, train acc:{:.4f}, test acc{:.4f}  \n".format(epoch, train_acc, test_acc_class))
            tqdm.write("epoch={}, train acc={:.4f}, test acc={:.4f} ".format(epoch, train_acc, test_acc_class))
            tqdm.write("best acc epoch={}, best test acc={:.4f} ".format(best_acc_epoch, best_test_acc))
            tqdm.write("")
             
            # lr scheduler
            # self.lr_scheduler.step(loss)
        f.write("best acc epoch={}, best test acc={:.4f} ".format(best_acc_epoch, best_test_acc))
        f.close()
           
   
    def test(self, test_loader):

        correct = 0.0   
        y_pred, y_true = [], [] 

        self.framework["model"].eval()
        self.framework["downstream_classifier"].eval()

        for _, (inputs, label, _, _, _) in enumerate(tqdm(test_loader, leave=False)):
            inputs = inputs.to(self.device, dtype=torch.float)
            label = label.to(self.device, dtype=torch.long)
            
            with torch.no_grad():
                feature = self.framework["model"](inputs)
                pred_class = self.framework["downstream_classifier"](feature)

            pred = pred_class.argmax(dim=1)
            for i in range(len(label)):
                y_pred.append(pred[i].item())
                y_true.append(label[i].item())
                if pred[i] == label[i]:
                    correct +=1

        test_acc_class = 100.*correct/len(test_loader.dataset)  
        return test_acc_class
    
    def save_framework(self, name):
        torch.save({
            "model": self.framework["model"].state_dict(),
            "downstream_classifier": self.framework["downstream_classifier"].state_dict(),
            "optimizer": self.optimizer,
        }, name)

    def load_framework(self, name):
        save = torch.load(name)
        self.framework["model"].load_state_dict(save["model"])
        self.framework["downstream_classifier"].load_state_dict(save["downstream_classifier"])