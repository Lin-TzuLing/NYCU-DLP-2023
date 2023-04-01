import numpy as np
# self import
from model_architecture.model import NNmodel

def MSEloss(pred_y, true_y):
    """loss function, mean square error"""
    loss = np.mean( (pred_y - true_y)**2 )
    return loss

def loss_derivative(pred_y, true_y):
    """loss derivative for backward"""
    return 2*(pred_y-true_y)/true_y.shape[0]

def accuracy(pred_y, true_y):
    """
    calculate accuracy
    turn pred_y into binary label with threshold 0.5
    """
    pred_label = np.where(pred_y > 0.5, 1, 0)
    acc = np.sum(np.equal(pred_label,true_y))/pred_y.shape[0]
    return pred_label, acc

def train(x, y, model, epoch, lr, report_every):
    loss_history, acc_history = [], []

    """training epoch"""
    for i in range(epoch):
        pred_y = model.forward(x)
        loss = MSEloss(pred_y, y)
        pred_label, acc = accuracy(pred_y, y)

        loss_history.append(loss)
        acc_history.append(acc)

        if acc==1.0:
            print('epoch {}, loss {:.4f}, acc {:.4f}'.format(i, loss, acc))
            break
        elif i%report_every ==0:
            print('epoch {}, loss {:.4f}, acc {:.4f}'.format(i, loss, acc))

        # backward propagation
        model.backward(loss_derivative(pred_y, y))
        model.update(lr=lr)

    return loss_history, acc_history, pred_label, pred_y

