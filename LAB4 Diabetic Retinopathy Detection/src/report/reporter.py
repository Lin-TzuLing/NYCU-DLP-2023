import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def write_result(args, train_loss, train_acc, test_acc):
    best_epoch, best_acc = 0, 0.0
    if args.pretrain_flag==True:
        name = "{}_{}_{}".format(args.model_type, "pretrained", str(args.learning_rate))
    else:
        name = "{}_{}_{}".format(args.model_type, "none", str(args.learning_rate))

    path = os.path.join(args.result_path,"statistic", name + ".txt")
    file = open(path, 'w')
    file.write("epoch \t train_loss \t train_acc \t test_acc \n")
    for i in range(len(train_loss)):
        file.write("{} \t {:.4f} \t {:.2f} \t {:.2f} \n".format(i, train_loss[i],
                                                                train_acc[i],
                                                                test_acc[i]))
        if test_acc[i]>best_acc:
            best_epoch = i
            best_acc = test_acc[i]
    file.write("best epoch {}, best acc = {:.2f}".format(best_epoch, best_acc))
    file.close()

def plot_history(args, train_loss, train_acc, test_acc, save_flag=False):
    """plot training loss, acc and test acc history"""
    if args.pretrain_flag==True:
        name = "{}_{}_{}".format(args.model_type, "pretrained", str(args.learning_rate))
    else:
        name = "{}_{}_{}".format(args.model_type, "none", str(args.learning_rate))

    """training loss history"""
    path = os.path.join(args.result_path, "pic")
    if args.pretrain_flag==True:
        plt.title("Pretrained {} Training Loss".format(args.model_type), fontsize=10)
    else:
        plt.title("{} Training Loss".format(args.model_type))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(np.arange(len(train_loss)), train_loss , 'b')
    if save_flag==True:
        plt.savefig(path+"/train_loss/{}.png".format(name))
    plt.show()
    plt.clf()

    """training acc history"""
    if args.pretrain_flag == True:
        plt.title("Pretrained {} Training Accuracy".format(args.model_type), fontsize=10)
    else:
        plt.title("{} Training Accuracy".format(args.model_type))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(np.arange(len(train_acc)), train_acc , 'r')
    if save_flag == True:
        plt.savefig(path + "/train_acc/{}.png".format(name))
    plt.show()
    plt.clf()

    """testing acc history"""
    if args.pretrain_flag == True:
        plt.title("Pretrained {} Test Accuracy".format(args.model_type), fontsize=10)
    else:
        plt.title("{} Test Accuracy".format(args.model_type))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(np.arange(len(test_acc)), test_acc , 'r')
    if save_flag == True:
        plt.savefig(path + "/test_acc/{}.png".format(name))
    plt.show()
    plt.clf()

def read_history(path):
    """read result"""
    file = open(path, "r")
    lines = file.read().split('\n')[1:-1]
    loss, train_acc, test_acc = [], [], []
    for line in lines:
        l = line.split('\t')
        loss.append(float(l[1]))
        # turn into %
        train_acc.append(float(l[2]))
        test_acc.append(float(l[3]))
    file.close()
    return loss, train_acc, test_acc

def plot_comparison(args):
    """plot comparison figure"""
    save_path = args.save_path
    name_pretrain = ("{}_{}_{}".format(args.model_type, "pretrained", str(args.learning_rate)))
    name_none = ("{}_{}_{}".format(args.model_type, "none", str(args.learning_rate)))
    path_pretrain = os.path.join(args.result_path, "statistic", name_pretrain + ".txt")
    path_none = os.path.join(args.result_path, "statistic", name_none + ".txt")
    loss_pretrain, train_acc_pretrain, test_acc_pretrain = read_history(path_pretrain)
    loss_none, train_acc_none, test_acc_none = read_history(path_none)

    """plot"""
    colors = [['r', 'orange'], ['g', 'b']]
    name_dict = {'resnet18':'ResNet18', 'resnet50':'ResNet50'}

    """acc"""
    plt.title('Result Comparison ({})'.format(name_dict[args.model_type]), fontsize=10)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy(%)")
    plt.plot(np.arange(len(train_acc_pretrain)), train_acc_pretrain,
             colors[0][0], label="Train(with pretraining)")
    plt.plot(np.arange(len(test_acc_pretrain)), test_acc_pretrain,
             colors[0][1], label="Test(with pretraining)")
    plt.plot(np.arange(len(train_acc_none)), train_acc_none,
             colors[1][0], label="Train(w/o pretraining)", marker=".")
    plt.plot(np.arange(len(test_acc_none)), test_acc_none,
             colors[1][1], label="Test(w/o pretraining)", marker=".")
    plt.legend(loc='upper left')
    plt.savefig(save_path + "/comparison_acc_{}.png".format(args.model_type))
    plt.show()
    plt.clf()


    """loss"""
    plt.title('Result Comparison ({})'.format(name_dict[args.model_type]), fontsize=10)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(np.arange(len(loss_pretrain)), loss_pretrain,
             colors[0][0], label="Train(with pretraining)")
    plt.plot(np.arange(len(loss_none)), loss_none,
             colors[0][1], label="Train(w/o pretraining)", marker=".")
    plt.legend(loc='upper left')
    plt.savefig(save_path + "/comparison_loss_{}.png".format(args.model_type))
    plt.show()
    plt.clf()



def plot_confusion(args, pred, label, num_classes):
    """plot confusion matrix"""
    save_path = args.save_path
    name_dict = {'resnet18': 'ResNet18', 'resnet50': 'ResNet50'}

    if args.pretrain_flag == True:
        name = "{}_{}".format(args.model_type, "pretrained")
    else:
        name = "{}_{}".format(args.model_type, "none")
    cm = confusion_matrix(label, pred, labels=np.arange(num_classes), normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title('Normalized Confusion Matrix ({})'.format(name_dict[args.model_type]), fontsize=10)
    plt.savefig(save_path + "/confusion_{}.png".format(name))
    plt.show()
    plt.clf()

