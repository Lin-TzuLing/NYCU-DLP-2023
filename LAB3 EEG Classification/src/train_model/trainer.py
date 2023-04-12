import os
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, lr_scheduler


class trainer():
    def __init__(self,args, model, device):

        self.args = args
        self.device = device
        self.model = model.to(device)
        self.lr = args.learning_rate
        self.epochs = args.epochs
        self.report_every = args.report_every
        self.save_path = os.path.join(args.save_path,
                                      args.model_type+"_"+args.activation_type+
                                      str(args.learning_rate)+".pt")


    def train(self, train_dataset, test_dataset):
        "batch-training model"

        train_loss_history, train_acc_history, test_acc_history = [], [], []
        total_train_num = len(train_dataset.dataset)
        best_epoch, best_acc = 0, 0.0

        if self.args.optimizer == "adam":
            optim = Adam(self.model.parameters(), lr=self.lr)
        loss_func = CrossEntropyLoss()
        if self.args.use_lrScheduler == True:
            scheduler = lr_scheduler.ReduceLROnPlateau(optim, factor=0.5,
                                                       patience=20, verbose=True,
                                                       mode='max', min_lr=1e-4, eps=1e-8)

        print(self.args.model_type+", "+str(self.args.learning_rate)+", "+self.args.activation_type)
        print("start training")
        for epoch in range(self.epochs):
            running_loss, running_train_correct = 0, 0

            """training"""
            self.model.train()
            for _, (train_data, train_label) in enumerate(train_dataset):
                pred_train = self.model(train_data)
                loss = loss_func(pred_train, train_label)

                loss.backward()
                optim.step()
                optim.zero_grad()

                running_loss += loss.item()
                running_train_correct += calculate_correct(pred_train, train_label)

            """training loss, acc"""
            train_loss = running_loss
            train_acc = running_train_correct/total_train_num

            """testing acc"""
            test_acc = self.test(test_dataset)

            """save model with highest test accuracy"""
            if test_acc > best_acc:
                torch.save(self.model.state_dict(), self.save_path)
                best_acc = test_acc
                best_epoch = epoch

            """append history"""
            train_loss_history.append(train_loss)
            train_acc_history.append(train_acc.item())
            test_acc_history.append(test_acc.item())

            """print loss and acc"""
            if epoch%self.report_every==0:
                print("epoch {}, total train loss = {:.4f}, "
                      "train acc = {:.4f}, test acc = {:.4f}".format(epoch, train_loss,
                                                                     train_acc, test_acc))

            """step lr scheduler based on test accuracy"""
            if self.args.use_lrScheduler == True:
                self.model.train()
                scheduler.step(test_acc)

        print("best epoch {}, test acc = {:.4f}".format(best_epoch, best_acc))
        print()
        return train_loss_history, train_acc_history, test_acc_history


    def test(self, test_dataset, valid_flag=True):
        # if valid_flag==False:
        #     self.model = self.model.load_state_dict(torch.load(self.save_path)).to(self.device)

        total_test_num = len(test_dataset.dataset)
        running_test_correct = 0

        self.model.eval()
        for _, (test_data, test_label) in enumerate(test_dataset):
            pred_test = self.model(test_data)
            running_test_correct += calculate_correct(pred_test, test_label)

        test_acc = running_test_correct/total_test_num
        return test_acc


def calculate_correct(pred, label):
    pred_label = torch.argmax(pred, 1, keepdim=False)
    return torch.sum(pred_label==label)
