import os
import torch
import torch.nn as nn
from tqdm import tqdm



class joint_trainer():
    def __init__(self, args, framework, device):
        super(joint_trainer).__init__()

        self.args = args
        self.lr = args.joint_lr
        self.epochs = args.joint_epochs
        self.device = device
        self.framework = framework
        self.classification_name = os.path.join(args.ckpt_path, "{}_{}_class.pth".format(self.args.arch,self.args.pretrained))
        self.keypoint_name = os.path.join(args.ckpt_path, "{}_{}_point.pth".format(self.args.arch,self.args.pretrained))
        if self.args.mode=='test':
            if self.args.task=='classification':
                self.load_framework(self.classification_name)
            elif self.args.task=='keypoint':
                self.load_framework(self.keypoint_name)

        # Loss function (train classfication model along with keypoint coord prediction)
        self.criterion_keypoint = nn.MSELoss()
        self.criterion_class = nn.CrossEntropyLoss()
    
        # Optimizer & Scheduler
        self.params = (list(self.framework["model"].parameters())+
                       list(self.framework["auxiliary_decoder"].parameters())+
                       list(self.framework["downstream_classifier"].parameters()))
        
        self.optimizer = torch.optim.SGD(self.params, lr=self.lr, momentum=0.9, weight_decay=5e-4)
        # self.optimizer = torch.optim.Adam(self.params, lr=self.lr, weight_decay=5e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                                       factor=0.7, patience=5, 
                                                                       min_lr=1e-7,  verbose=True)
        

    def train(self, train_loader, test_loader):        
        best_loss_epoch, best_acc_epoch = 0, 0 
        best_test_loss, best_test_acc = 100000.0, 0.0
        y_pred, y_true = [], []
        f = open(f"{self.args.record_path}/{self.args.arch}_{self.args.pretrained}_joint.txt", "w")
        
        # training epochs
        for epoch in tqdm(range(self.epochs)):
            # training all network
            self.framework["model"].train()
            self.framework["auxiliary_decoder"].train()
            self.framework["downstream_classifier"].train()

            total_loss_keypoint, total_loss_class, correct = 0.0, 0.0, 0.0

            # training batch
            for _, (inputs, label, point, _, _) in enumerate(tqdm(train_loader, leave=False)):
                inputs = inputs.to(self.device, dtype=torch.float)
                label = label.to(self.device, dtype=torch.long)
                points = point.to(self.device, dtype=torch.float).view(-1, point.shape[1]*point.shape[2])
                
                feature = self.framework["model"](inputs)
                pred_keypoint = self.framework["auxiliary_decoder"](feature)
                pred_class = self.framework["downstream_classifier"](feature)

                if (label.shape)[0]==1:
                    pred_class = pred_class.unsqueeze(0)

                loss_keypoint = self.criterion_keypoint(pred_keypoint, points)
                loss_class = self.criterion_class(pred_class, label)

                total_loss_keypoint += loss_keypoint.item()
                total_loss_class += loss_class.item()

                # jointly train two task
                loss = loss_keypoint + loss_class

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
            train_loss_keypoint = total_loss_keypoint/len(train_loader)
            train_acc = 100.*correct/len(train_loader.dataset)

            # evaluation
            test_loss_keypoint, test_acc_class = self.test(test_loader)
            

            # save model
            if test_loss_keypoint < best_test_loss:
                best_test_loss = test_loss_keypoint
                best_loss_epoch = epoch
                self.save_framework(self.keypoint_name)
                tqdm.write("Best Keypoint Model saved!")  
            if test_acc_class > best_test_acc:
                best_test_acc = test_acc_class
                best_acc_epoch = epoch
                self.save_framework(self.classification_name)
                tqdm.write("Best Classification Model saved!")  
            
            
            # write records
            f.write("epoch={}, train loss={:.4f}, test loss={:.4f}, train acc={:.4f}, test acc={:.4f}  \n".format(epoch, train_loss_keypoint, test_loss_keypoint,  train_acc, test_acc_class))
            tqdm.write("epoch={}, train loss={:.4f}, test loss={:.4f}, train acc={:.4f}, test acc={:.4f} ".format(epoch, train_loss_keypoint, test_loss_keypoint, train_acc, test_acc_class))
            tqdm.write("best loss epoch={}, best acc epoch={}, best test loss={:.4f}, best test acc={:.4f} ".format(best_loss_epoch, best_acc_epoch, best_test_loss, best_test_acc))
            tqdm.write("")
             
            # lr scheduler
            # self.lr_scheduler.step(loss)
        f.write("best loss epoch={}, best acc epoch={}, best test loss={:.4f}, best test acc={:.4f} ".format(best_loss_epoch, best_acc_epoch, best_test_loss, best_test_acc))
        f.close()
           
   
    def test(self, test_loader):

        total_loss_keypoint = 0.0
        correct = 0.0   
        y_pred, y_true = [], [] 

        self.framework["model"].eval()
        self.framework["auxiliary_decoder"].eval()
        self.framework["downstream_classifier"].eval()

        for _, (inputs, label, point, _, _) in enumerate(tqdm(test_loader, leave=False)):
            inputs = inputs.to(self.device, dtype=torch.float)
            label = label.to(self.device, dtype=torch.long)
            points = point.to(self.device, dtype=torch.float).view(-1, point.shape[1]*point.shape[2])
            
            with torch.no_grad():
                feature = self.framework["model"](inputs)
                pred_keypoint = self.framework["auxiliary_decoder"](feature)
                pred_class = self.framework["downstream_classifier"](feature)

            loss_keypoint = self.criterion_keypoint(pred_keypoint, points)
            total_loss_keypoint += loss_keypoint.item()

            pred = pred_class.argmax(dim=1)
            for i in range(len(label)):
                y_pred.append(pred[i].item())
                y_true.append(label[i].item())
                if pred[i] == label[i]:
                    correct +=1

        test_loss_keypoint = total_loss_keypoint/len(test_loader)
        test_acc_class = 100.*correct/len(test_loader.dataset)  
        return test_loss_keypoint, test_acc_class
    

    def save_framework(self, name):
        torch.save({
            "model": self.framework["model"].state_dict(),
            "auxiliary_decoder": self.framework["auxiliary_decoder"].state_dict(),
            "downstream_classifier": self.framework["downstream_classifier"].state_dict(),
            "optimizer": self.optimizer,
        }, name)

    def load_framework(self, name):
        save = torch.load(name)
        self.framework["model"].load_state_dict(save["model"])
        self.framework["auxiliary_decoder"].load_state_dict(save["auxiliary_decoder"])
        self.framework["downstream_classifier"].load_state_dict(save["downstream_classifier"])