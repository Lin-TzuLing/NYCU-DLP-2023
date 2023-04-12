import os
import numpy as np
import matplotlib.pyplot as plt


def write_result(args, train_loss, train_acc, test_acc):
    best_epoch, best_acc = 0, 0.0
    path = os.path.join(args.result_path,"statistic",
                        args.model_type + "_" + args.activation_type + "_"+
                        str(args.learning_rate) + ".txt")
    file = open(path, 'w')
    file.write("epoch \t train_loss \t train_acc \t test_acc \n")
    for i in range(len(train_loss)):
        file.write("{} \t {:.4f} \t {:.4f} \t {:.4f} \n".format(i, train_loss[i],
                                                                train_acc[i],
                                                                test_acc[i]))
        if test_acc[i]>best_acc:
            best_epoch = i
            best_acc = test_acc[i]
    file.write("best epoch {}, best acc = {:.4f}".format(best_epoch, best_acc))
    file.close()

def plot_history(args, train_loss, train_acc, test_acc, save_flag=False):
    """plot training loss, acc and test acc history"""

    """training loss history"""
    path = os.path.join(args.result_path, "pic")
    plt.title('Training Loss (CrossEntropy)', fontsize=10)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("{} Loss Curve".format(args.model_type))
    plt.plot(np.arange(len(train_loss)),train_loss, 'b')
    if save_flag==True:
        plt.savefig(path+"/train_loss/TrainLoss_{}_{}_{}.png".format(args.model_type,
                                                          args.activation_type,
                                                          str(args.learning_rate)))
    plt.show()
    plt.clf()

    """training acc history"""
    plt.title('Training Accuracy', fontsize=10)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("{} Accuracy Curve".format(args.model_type))
    plt.plot(np.arange(len(train_acc)), train_acc, 'r')
    if save_flag == True:
        plt.savefig(path + "/train_acc/TrainAcc_{}_{}_{}.png".format(args.model_type,
                                                            args.activation_type,
                                                            str(args.learning_rate)))
    plt.show()
    plt.clf()

    """testing acc history"""
    plt.title('Test Accuracy', fontsize=10)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("{} Accuracy Curve".format(args.model_type))
    plt.plot(np.arange(len(test_acc)), test_acc, 'r')
    if save_flag == True:
        plt.savefig(path + "/test_acc/TestAcc_{}_{}_{}.png".format(args.model_type,
                                                           args.activation_type,
                                                           str(args.learning_rate)))
    plt.show()
    plt.clf()



def read_history(path):
    """read result"""
    file = open(path, "r")
    lines = file.read().split('\n')[1:-1]
    loss, train_acc, test_acc, best_acc = [], [], [], []
    for line in lines:
        l = line.split('\t')
        loss.append(float(l[1]))
        # turn into %
        train_acc.append(float(l[2])*100)
        test_acc.append(float(l[3])*100)
    file.close()
    return loss, train_acc, test_acc

def plot_comparison(train_loss_dict, train_acc_dict, test_acc_dict, save_path, models, lr):
    """plot test accuracy comparison of different activation function"""
    activations = ['relu', 'leaky_relu', 'elu']
    colors = [['b','orange'],['limegreen','r'],['darkviolet','hotpink']]
    for model in models:
        if model=='EEGNet':
            model_nickname = 'eeg'
        elif model=='DeepConvNet':
            model_nickname = 'deepconv'
        elif model=='VGGNet':
            model_nickname = 'vgg'
        else:
            raise ValueError('wrong model name')

        """plot train/test accuracy comparison"""
        plt.title('Activation function comparison ('+model+')', fontsize=10)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy(%)")
        for i in range(len(activations)):
            activation = activations[i]
            train_acc = train_acc_dict[model_nickname + "_" + activation + "_" + str(lr)]
            test_acc = test_acc_dict[model_nickname + "_" + activation + "_" + str(lr)]
            plt.plot(np.arange(len(train_acc)), train_acc, colors[i][0], label=activation+"_train")
            plt.plot(np.arange(len(test_acc)), test_acc, colors[i][1], label=activation+"_test")

        plt.yticks(np.arange(50, 105, step=5))
        plt.legend(loc='lower right')
        plt.savefig(save_path + "/acc/{}_{}.png".format(model, str(lr)))
        plt.show()
        plt.clf()

        """plot train loss comparison"""
        plt.title('Activation function comparison (' + model + ')', fontsize=10)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        for i in range(len(activations)):
            activation = activations[i]
            train_loss = train_loss_dict[model_nickname + "_" + activation + "_" + str(lr)]
            plt.plot(np.arange(len(train_loss)), train_loss, colors[i][0], label=activation + "_train")
        plt.legend(loc='lower right')
        plt.savefig(save_path + "/loss/{}_{}.png".format(model, str(lr)))
        plt.show()
        plt.clf()



def plot_activation(save_path):
    """plot activation functions"""
    values = np.arange(-10, 10, 0.1)

    """plot ReLU and its derivative"""
    plt.title('ReLU function')
    plt.plot(values, relu(values), label='ReLU', color='blue')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='lower right')
    plt.savefig(save_path + "/relu.png")
    plt.show()
    plt.clf()

    plt.title('ReLU derivative')
    plt.plot(values, relu_derivative(values), label='ReLU derivative', color='orange')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='lower right')
    plt.savefig(save_path + "/relu_derivative.png")
    plt.show()
    plt.clf()

    """plot Leaky ReLU and its derivative"""
    leaky_alpha = 0.05
    plt.title('Leaky ReLU function  (alpha = '+str(leaky_alpha)+')')
    plt.plot(values, leaky_relu(values, alpha=leaky_alpha),
             label='Leaky ReLU, alpha='+str(leaky_alpha), color='blue')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='lower right')
    plt.savefig(save_path + "/leaky_relu.png")
    plt.show()
    plt.clf()

    plt.title('Leaky ReLU derivative  (alpha = ' + str(leaky_alpha) + ')')
    plt.plot(values, leaky_relu_derivative(values, alpha=leaky_alpha),
             label='Leaky ReLU derivative, alpha=' + str(leaky_alpha), color='orange')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='lower right')
    plt.savefig(save_path + "/leaky_relu_derivative.png")
    plt.show()
    plt.clf()

    """plot ELU and its derivative"""
    elu_alpha = 1
    plt.title('ELU function  (alpha = ' + str(elu_alpha) + ')')
    plt.plot(values, elu(values,alpha=elu_alpha),
             label='ELU, alpha='+str(elu_alpha), color='blue')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='lower right')
    plt.savefig(save_path + "/elu.png")
    plt.show()
    plt.clf()

    plt.title('ELU derivative  (alpha = ' + str(elu_alpha) + ')')
    plt.plot(values, elu_derivative(values, alpha=elu_alpha),
             label='ELU derivative, alpha=' + str(elu_alpha), color='orange')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='lower right')
    plt.savefig(save_path + "/elu_derivative.png")
    plt.show()
    plt.clf()

def relu(x):
    # [if>0, return x]; [if<=0, return 0]
    return [max(0,value) for value in x]

def relu_derivative(x):
    # [if>0, return 1]; [if<=0, return 0]
    return np.where(x > 0, 1, 0)

def leaky_relu(x, alpha):
    # [if>0, return x]; [if<=0, return (alpha * x)]
    return np.where(x > 0, x, alpha*x)

def leaky_relu_derivative(x, alpha):
    # [if>0, return 1]; [if<=0, return (alpha)]
    return np.where(x > 0, 1, alpha)

def elu(x, alpha):
    # [if>0, return x]; [if<=0, return (alpha * exp(x) - 1)]
    return np.where(x > 0, x, alpha*(np.exp(x)-1))

def elu_derivative(x, alpha):
    # [if>0, return 1]; [if<=0, return (alpha * exp(x))]
    return np.where(x > 0 , 1, alpha * np.exp(x))


def rewrite_best_result(path, save_path, model, activation, lr):
    """summarize best test accuracy of all model and lr combinations"""
    file = open(path, "r")
    lines = file.read().split('\n')[-1]
    l = lines.split('=')
    best_epoch = l[0].split(' ')[2]
    best_acc = float(l[1])*100
    file.close()
    summary_file = open(save_path, "a")
    message = model+','+activation+','+lr+'\t'+best_epoch+'\t'+str(best_acc)+'% \n'
    print('model = {}, activation = {}, lr = {}, highest accuracy = {:.2f}%'.format(model,
                                                                                    activation,
                                                                                    lr, best_acc))
    summary_file.write(message)
    summary_file.close()




if __name__=="__main__":

    """plot comparison figure"""
    models = ["eeg", "deepconv", "vgg"]
    activations = ["relu", "leaky_relu", "elu"]
    lrs = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4]
    train_loss_dict, train_acc_dict, test_acc_dict = {}, {}, {}
    save_path = '../../result/statistic/all.txt'
    summary_file = open(save_path, "w")
    summary_file.write('model \t best epoch \t best acc (%) \n')
    summary_file.close()
    # read all history and store in dict
    for model in models:
        for activation in activations:
            for lr in lrs:
                path = os.path.join('../../result/statistic/', model+"_"+activation+"_"+str(lr)+".txt")
                loss, train_acc, test_acc = read_history(path=path)
                train_loss_dict[model+"_"+activation+"_"+str(lr)] = loss
                train_acc_dict[model + "_" + activation + "_" + str(lr)] = train_acc
                test_acc_dict[model + "_" + activation + "_" + str(lr)] = test_acc
                # also summarize best test acc result in this file,
                # rewrite it into a new file
                rewrite_best_result(path=path, save_path=save_path, model=model,
                                    activation=activation, lr=str(lr))

    # plot Activation Comparison figure (with certain learning rate target_lr)
    target_models = ["EEGNet", "DeepConvNet", "VGGNet"]
    target_lr = float(2e-3)
    save_path = '../../result/pic/comparison'
    plot_comparison(train_loss_dict, train_acc_dict, test_acc_dict, save_path, target_models, target_lr)
    print('plot comparison done')

    """plot activation function figures"""
    save_path = '../../result/pic'
    plot_activation(save_path=save_path)
    print('plot activations done')