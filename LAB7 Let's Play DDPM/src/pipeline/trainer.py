import numpy as np
import os
import torch
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
# self-defined import
from utils.evaluator import evaluation_model


class trainer():
    def __init__(self, args, ddpm, device):
        super(trainer).__init__()

        self.args = args
        self.lr = args.lr
        self.n_epoch = args.n_epoch
        self.device = device
        self.ddpm = ddpm    
        self.evaluator = evaluation_model()
        

    def train(self, train_loader, test_loader, test_new_loader):

        optim = torch.optim.Adam(self.ddpm.parameters(), lr=self.lr)
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', 
        #                                                           factor=0.5, patience=10, 
        #                                                           min_lr=1e-6,  verbose=True)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', 
                                                                  factor=0.95, patience=5, 
                                                                  min_lr=0,  verbose=True)
        best_test_epoch, best_test_new_epoch = 0, 0
        best_test_score, best_test_new_score = 0, 0

        for epoch in tqdm(range(self.n_epoch)):
            tqdm.write("epoch {}".format(epoch))
            
            """training time"""
            self.ddpm.train()
            optim.param_groups[0]['lr'] = self.lr*(1-epoch/self.n_epoch) # linear lr decay
            # optim.param_groups[0]['lr'] = max(0, optim.param_groups[0]['lr']-self.lr*(1/self.n_epoch))
            # optim.param_groups[0]['lr'] = max(0, optim.param_groups[0]['lr']-self.lr*(1e-4/self.n_epoch))
            tqdm.write("lr: {}".format(optim.param_groups[0]['lr']))

            pbar = tqdm(train_loader, leave=False)
            loss_ema = None
            for x, cond in pbar:
                optim.zero_grad()
                x = x.to(self.device) # [bs,3,64,64]
                cond = cond.to(self.device) # [bs,24]
                loss = self.ddpm(x, cond)
                loss.backward()
                if loss_ema is None:
                    loss_ema = loss.item()
                else:
                    loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
                pbar.set_description(f"loss: {loss_ema:.4f}")
                optim.step()
            
            """testing time"""
            test_score, grid = self.test(test_loader)
            path = os.path.join(self.args.result_path, "test_{}.png".format(epoch))
            save_image(grid, path)

            test_new_score, grid = self.test(test_new_loader)
            path = os.path.join(self.args.result_path, "test_new_{}.png".format(epoch))
            save_image(grid, path)

            if test_score > best_test_score:
                best_test_score = test_score
                best_test_epoch = epoch
                path = os.path.join(self.args.model_path, self.args.noise_type, "model_test.pth")
                torch.save(self.ddpm.state_dict(), path)
                tqdm.write("saved test model")

            if test_new_score > best_test_new_score:
                best_test_new_score = test_new_score
                best_test_new_epoch = epoch
                path = os.path.join(self.args.model_path, self.args.noise_type, "model_test_new.pth")
                torch.save(self.ddpm.state_dict(), path)
                tqdm.write("saved test new model")
                           
            tqdm.write('test:{:.4f}, new test:{:.4f}'.format(test_score, test_new_score))
            tqdm.write('best test epoch:{}, best new test epoch:{}'.format(best_test_epoch, best_test_new_epoch))
            tqdm.write('best test score:{:.4f}, best new test score:{:.4f}'.format(best_test_score, best_test_new_score))
            
            # save training model
            path = os.path.join(self.args.model_path, self.args.noise_type, "model_latest_train.pth")
            torch.save(self.ddpm.state_dict(), path)

            # lr scheduler step based on best_test_score + best_test_new_score
            lr_scheduler.step(best_test_score+best_test_new_score)
        
    def test(self, test_loader):
        self.ddpm.eval()
        pbar1 = tqdm(test_loader, leave=False)
        x_gen, label = [], []
        with torch.no_grad():
            for cond in pbar1:
                cond = cond.to(self.device)
                x_i = self.ddpm.sample(cond, (3, 64, 64), self.device)
                x_gen.append(x_i)
                label.append(cond)
                # x_gen.append(x_i.detach().cpu().numpy())
            x_gen = torch.stack(x_gen, dim=0).squeeze()
            label = torch.stack(label, dim=0).squeeze()
            score = self.evaluator.eval(x_gen, label)
            grid = make_grid(x_gen, nrow=8, normalize=True)
                
        return score, grid
    


    