import torch
import torch.nn as nn
from torch.optim import SGD



class trainer():
    def __init__(self, args, model, device):
        super(trainer, self).__init__()
        self.args = args
        self.model = model.to(device)
        self.device = device
        if self.args.pretrain_flag==True:
            self.model_name = "{}_pretrain_{}.pt".format(self.args.model_type,
                                                        self.args.learning_rate)
        else:
            self.model_name = "{}_none_{}.pt".format(self.args.model_type,
                                                        self.args.learning_rate)
    def train(self, train_dataset, test_dataset):
        """training model"""

        """set loss and optimizer"""
        loss_func = nn.CrossEntropyLoss()
        optimizer = SGD(self.model.parameters(), lr=self.args.learning_rate,
                        momentum=0.9, weight_decay=5e-4)

        """save loss and acc history"""
        train_loss_history, train_acc_history, test_acc_history = [], [], []
        best_epoch, best_acc = 0, 0

        """batch training"""
        print("model: {}, pretrain:{}, lr: {}, bs:{}".format(self.args.model_type,
                                                      self.args.pretrain_flag,
                                                      self.args.learning_rate,
                                                      self.args.batch_size))
        print("start training")
        for epoch in range(self.args.epochs):
            running_loss, running_train_correct = 0, 0

            """feature extract before finetune"""
            if epoch == 5:
                self.model.set_param_requires_grad(self.model, feature_extract=False)
                optimizer = SGD(self.model.parameters(), lr=self.args.learning_rate,
                                momentum=0.9, weight_decay=5e-4)

            self.model.train()
            for _, (train_data, train_label) in enumerate(train_dataset):
                train_data = train_data.to(self.device)
                train_label = train_label.to(self.device)
                pred_train = self.model(train_data)
                loss = loss_func(pred_train, train_label.long())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                running_loss += loss.item()
                running_train_correct += calculate_correct(pred_train, train_label)

            """train loss/acc and test acc"""
            train_loss = running_loss
            train_acc = (running_train_correct/len(train_dataset.dataset))*100
            test_acc = (self.test(test_dataset))*100

            """save model with highest test accuracy"""
            if test_acc > best_acc:
                torch.save(self.model.state_dict(), self.args.model_path+self.model_name)
                best_acc = test_acc
                best_epoch = epoch

            """append history"""
            train_loss_history.append(train_loss)
            train_acc_history.append(train_acc.item())
            test_acc_history.append(test_acc.item())

            """print loss and acc"""
            print("epoch {}, total train loss = {:.4f}, "
                  "train acc = {:.2f}%, test acc = {:.2f}%".format(epoch, train_loss,
                                                                 train_acc, test_acc))

        print("best epoch {}, test acc = {:.2f}".format(best_epoch, best_acc))
        print()
        return train_loss_history, train_acc_history, test_acc_history


    def test(self, test_dataset):
        running_test_correct = 0
        self.model.eval()
        for _, (test_data, test_label) in enumerate(test_dataset):
            test_data = test_data.to(self.device)
            test_label = test_label.to(self.device)
            pred_test = self.model(test_data)
            running_test_correct += calculate_correct(pred_test, test_label)

        test_acc = running_test_correct/len(test_dataset.dataset)
        return test_acc

    def demo(self, test_dataset):
        print("model: {}, pretrain:{}, lr: {}, bs:{}".format(self.args.model_type,
                                                             self.args.pretrain_flag,
                                                             self.args.learning_rate,
                                                             self.args.batch_size))
        print("start demo")
        print("loading model weight")
        self.model.load_state_dict(torch.load(self.args.model_path+self.model_name))

        pred, label = [], []
        running_test_correct = 0
        self.model.eval()
        for _, (test_data, test_label) in enumerate(test_dataset):
            test_data = test_data.to(self.device)
            test_label = test_label.to(self.device)
            pred_test = self.model(test_data)
            pred_label = torch.argmax(pred_test, dim=1, keepdim=False)
            running_test_correct += calculate_correct(pred_test, test_label)
            pred += pred_label.tolist()
            label += test_label.tolist()

        test_acc = running_test_correct / len(test_dataset.dataset)* 100
        return pred, label, test_acc.item()


def calculate_correct(pred, label):
    pred_label = torch.argmax(pred, dim=1, keepdim=False)
    return torch.sum(pred_label == label)
