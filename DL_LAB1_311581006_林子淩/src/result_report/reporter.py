import matplotlib.pyplot as plt
import os
import csv
import numpy as np

def show_result(args, x, y, pred_y):
    plt.subplot(1,2,1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i]==0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.show()
    plt.savefig("prediction_{}.png".format(args.data_type))


def save_result(args, best_epoch, best_acc):
     with open(args.exp_path+".csv", 'a') as file:
        writer = csv.writer(file)
        writer.writerow(['Data Type', 'activation', 'hidden_dim',
                         'lr', 'best epoch', 'best_acc'])
        for i in range(args.exp_iterations):
            writer.writerow([args.data_type, args.activation_type, args.hidden_dim,
                             args.lr, best_epoch[i], best_acc[i]])
        writer.writerow([])


def show_history_plot(args, loss_history, acc_history):
    # plt.subplot(6, 6, 1)
    plt.title('Loss history curve (MSE)', fontsize=10)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("{} Loss Curve".format(args.data_type))
    plt.plot(np.arange(len(loss_history)),loss_history, 'b')
    plt.show()
    plt.savefig("historyLoss_{}.png".format(args.data_type))
    plt.clf()
    # plt.subplot(6, 6, 2)
    plt.title('Accuracy history curve', fontsize=10)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("{} Accuracy Curve".format(args.data_type))
    plt.plot(np.arange(len(acc_history)), acc_history, 'b')
    plt.show()
    plt.savefig("historyAcc_{}.png".format(args.data_type))



def plot_sigmoid():
    values = np.arange(-10, 10, 0.1)
    plt.plot(values, sigmoid(values), label='sigmoid', color='blue')
    plt.plot(values, sigmoid_derivative(values), label='sigmoid derivative', color='orange')
    plt.xlabel('x')
    plt.legend()
    plt.show()

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def plot_relu():
    values = np.arange(-10, 10, 0.1)
    plt.plot(values, relu(values), label='ReLU', color='blue')
    plt.plot(values, relu_derivative(values), label='ReLU derivative', color='orange')
    plt.xlabel('x')
    plt.legend()
    plt.show()

def relu(x):
    return [max(0,value) for value in x]

def relu_derivative(x):
    return (x > 0) * 1

if __name__=="__main__":
    plot_sigmoid()
    plot_relu()