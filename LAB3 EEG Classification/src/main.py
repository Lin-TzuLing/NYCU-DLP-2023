import numpy as np
import random
import torch
import argparse

# self-import
from data_loader.dataloader import read_bci_data, create_dataset
from model_architecture.model import build_model
from train_model.trainer import trainer
from report.reporter import write_result, plot_history

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../data/",
                        help="path to train/test data")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="training batchsize")
    parser.add_argument("--model_type", type=str, default="eeg",
                        choices=["eeg", "deepconv", "vgg"],
                        help="model type")
    parser.add_argument("--activation_type", type=str, default="relu",
                        choices=["relu", "leaky_relu", "elu"],
                        help="activation function type")
    parser.add_argument("--dropout_ratio", type=float, default=0.25,
                        help="EEGNet dropout ratio")
    parser.add_argument("--epochs", type=int, default=500,
                        help="training epoch num")
    parser.add_argument("--learning_rate", type=float, default=2e-3,
                        choices=[1e-2, 5e-3, 2.5e-3, 2e-3, 1e-3, 5e-4, 2.5e-4, 2e-4, 1e-4],
                        help="learning rate")
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "adamW"],
                        help="optimizer")
    parser.add_argument("--report_every", type=int, default=10,
                        help="report every {} epoch")
    parser.add_argument("--save_path", type=str, default="../save_model",
                        help="save model weight")
    parser.add_argument("--result_path", type=str, default="../result",
                        help="save result")
    parser.add_argument("--use_lrScheduler", type=bool, default=False,
                        help = "use ReduceLROnPlateau")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed")
    args = parser.parse_args()
    return args


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:{}'.format(device))

    """get params"""
    args = arg_parse()
    # args.seed = 10
    # args.use_lrScheduler = True
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # TODO
    # args.model_type = "vgg"
    # args.activation_type = "relu"
    # args.batch_size = 64

    """get data"""
    train_data, train_label, test_data, test_label = read_bci_data(args)
    train_dataset, test_dataset = create_dataset(args, device,
                                                 train_data, train_label,
                                                 test_data, test_label)


    """get model"""
    model = build_model(args)
    model.initialize_weights()

    """start training"""
    trainer = trainer(args, model, device)
    train_loss_history, train_acc_history, test_acc_history = trainer.train(train_dataset,
                                                                                test_dataset)
    """write result to txt"""
    write_result(args, train_loss_history, train_acc_history, test_acc_history)

    """plot history curve"""
    plot_history(args, train_loss_history, train_acc_history, test_acc_history, save_flag=False)
