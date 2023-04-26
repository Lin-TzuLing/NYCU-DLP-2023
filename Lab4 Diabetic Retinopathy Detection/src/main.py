import torch
import random
import numpy as np
import argparse
from torchinfo import summary
import warnings

from data_loader.dataloader import create_dataset
from model_architecture.model import build_model
from train_model.trainer import trainer
from report.reporter import write_result, plot_history, plot_comparison, plot_confusion

def arg_parse():
    parser = argparse.ArgumentParser()
    # demo
    parser.add_argument("--demo", action='store_true',
                        help="demo")
    # dataset
    parser.add_argument("--data_path", type=str, default="../data/",
                        help="path to train/test_img.csv and train/test_label.csv")
    parser.add_argument("--dataset_path", type=str, default="../data/",
                        help="path to new_train/new_test (root)")
    parser.add_argument("--mode", type=str, default="train",
                        choices=['train', 'test'],
                        help="path to train/test data")
    parser.add_argument("--batch_size", type=int, default=6,
                        help="batch size for training and testing")
    parser.add_argument("--normalize", action='store_true',
                        help="normalize data or not")
    # model
    parser.add_argument("--model_type", type=str, default="resnet50",
                        choices=['resnet18', 'resnet50'],
                        help="choose model type")
    parser.add_argument("--pretrain_flag", action='store_true',
                        help="load pretrained weight or not")
    parser.add_argument("--seed", type=int, default=123,
                        help="fixed random seed")

    # training
    parser.add_argument("--epochs", type=int, default=10,
                        help="number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        choices=[1e-3, 5e-4],
                        help="learning rate")
    parser.add_argument("--optimizer", type=str, default="sgd",
                        choices=['sgd'],
                        help="optimizer type")
    # save path
    parser.add_argument("--model_path", type=str, default="../model_weight/",
                        help="path to save trained model weight")
    parser.add_argument("--result_path", type=str, default="../result/",
                        help="path to save statistic result")
    parser.add_argument("--save_path", type=str, default="../result/pic",
                        help="path to save pic result")
    args = parser.parse_args()
    return args


if __name__=="__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    """check cuda"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:{}'.format(device))


    """get param"""
    args = arg_parse()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # demo
    args.pretrain_flag = True
    args.demo = True
    args.batch_size = 12
    args.model_type = "resnet18"

    # args.pretrain_flag = False
    # args.demo = True
    # args.batch_size = 12
    # args.model_type = "resnet18"

    # args.pretrain_flag = True
    # args.demo = True
    # args.batch_size = 6
    # args.model_type = "resnet50"

    # args.pretrain_flag = False
    # args.demo = True
    # args.batch_size = 6
    # args.model_type = "resnet50"

    """get data"""
    train_dataset, test_dataset, num_classes = create_dataset(args, device)

    """build model"""
    model = build_model(args, num_classes)
    model.initialize_weights()
    summary(model, input_size=(args.batch_size, 3, 512, 512))

    """build trainer"""
    trainer = trainer(args, model, device)
    if args.demo:
        pred, label, test_acc = trainer.demo(test_dataset)
        print("test acc = {:.2f}%".format(test_acc))
        plot_confusion(args, pred, label, num_classes)
        plot_comparison(args)
        print()
    else:
        train_loss_history, train_acc_history, test_acc_history = trainer.train(train_dataset,
                                                                                test_dataset)
        """write result to txt"""
        write_result(args, train_loss_history, train_acc_history, test_acc_history)

        # """plot history curve"""
        plot_history(args, train_loss_history, train_acc_history, test_acc_history, save_flag=True)
        # print()
