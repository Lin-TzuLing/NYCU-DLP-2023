import numpy as np
from argparse import ArgumentParser

# self import
from data.dataLoader import load_data
from model_architecture.model import NNmodel
from model_architecture.trainer import train
from result_report.reporter import show_result, save_result, show_history_plot

def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("-exp_path", type=str, default="",
                        help="csv path for saving result")
    parser.add_argument("-exp_iterations", type=int, default=1,
                        help="how many iterations for one experiment")
    parser.add_argument("-data_type", type=str, default="linear",
                        choices=["linear", "xor"], help="data type")
    parser.add_argument("-activation_type", type=str, default="sigmoid",
                        choices=["none", "sigmoid", "relu"], help="activation type")
    parser.add_argument("-hidden_dim", type=int, default=50,
                        help="hidden units of NN network")
    parser.add_argument("-train_epoch", type=int, default=100000,
                        help="training epoch number")
    parser.add_argument("-lr", type=float, default=1e-1,
                        help="learning rate")
    parser.add_argument("-report_every", type=int, default=5000,
                        help="print loss and acc every () epochs")
    args = parser.parse_args()
    return args

if __name__=="__main__":
    """get param"""
    args = arg_parser()
    # args.data_type="xor"

    """set random seed"""
    np.random.seed(1)

    """load data"""
    x, y = load_data(data_type=args.data_type)


    """training process"""
    best_epoch = []
    best_acc = []
    for i in range(args.exp_iterations):
        """build model"""
        # x[100,2] -> [100,hidden] -> pred_y[100,1] <-> true_y[100,1]
        model = NNmodel(in_dim=x.shape[1],
                        hidden_dim=args.hidden_dim,
                        out_dim=1,
                        data_type=args.data_type.lower(),
                        activation_type=args.activation_type.lower())

        """start training"""
        print('exp_iteration {}'.format(i))
        loss_history, acc_history, pred_label, pred_y = train(x,y,model, args.train_epoch,
                                                              args.lr, args.report_every)
        best_epoch.append(len(acc_history))
        best_acc.append(acc_history[-1])

    if args.exp_path!="":
        save_result(args, best_epoch, best_acc)
    else:
        # show pred_label instead of pred_y
        show_result(args, x, y, pred_label)
        show_history_plot(args, loss_history, acc_history)
        # print pred_y
        if args.data_type=='linear':
            print(pred_y.reshape((25,4)))
        else:
            print(pred_y)
    print('done')