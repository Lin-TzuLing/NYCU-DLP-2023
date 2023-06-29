import argparse
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# import models
import torch
import warnings
warnings.filterwarnings("ignore")

from data.dataloader import load_data
from models.build_model import build_model
from pipeline.joint_trainer import joint_trainer
from pipeline.baseline_trainer import baseline_trainer



    

def arg_parse():
    parser = argparse.ArgumentParser(description='Neural Network')
    parser.add_argument('-b', '--batchsize', default=4, type=int, help='batch size')
    parser.add_argument('--img_size', default=(512,512), nargs='+', type=int)
    parser.add_argument('--joint_lr', default=1e-3, type=float, 
                        help='Learning rate for downstream classfication')
    parser.add_argument('--joint_epochs', default=150, type=int, 
                        help='# downstream classfication finetune epochs')
    parser.add_argument('--arch', default='resnet18', type=str, help='resnet50 or resnet18')
    parser.add_argument('--mode', default='train', choices=["train", "test", "baseline_train", "baseline_test"],
                        type=str, help='train or test meta model, or run baseline')
    parser.add_argument('--task', default='classification', 
                        choices=["classification", "keypoint"],
                        type=str, help='task for testing, pose classification or keypoint coordination prediction')
    parser.add_argument('--pretrained', action="store_true", help='pretrained or not')
    parser.add_argument('--train_path', default="/home/lin/Desktop/dlp_final/DATASET/TEST", 
                        type=str, help="training dataset path")
    parser.add_argument('--test_path', default="/home/lin/Desktop/dlp_final/DATASET/TRAIN", 
                        type=str, help="testing dataset path")
    parser.add_argument('--ckpt_path', default="/home/lin/Desktop/dlp_final/result_ckpt", 
                        type=str, help="save model ckpt path")
    parser.add_argument('--record_path', default="/home/lin/Desktop/dlp_final/result_record", 
                        type=str, help="save model ckpt path")
    parser.add_argument('--num_workers', default=4 ,type=int)
    parser.add_argument('--seed', default=10 ,type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = arg_parse()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print("Mode:" , args.mode)
    print("Model:", args.arch)
    print("Pretrained:", args.pretrained)

    train_loader, test_loader, n_classes, n_keypoint_coord = load_data(args, device)
    framework = build_model(args, n_classes, n_keypoint_coord, device)

    if args.mode=='train':
        trainer = joint_trainer(args, framework, device)
        trainer.train(train_loader, test_loader)
        print("done Meta model training")

    elif args.mode=='test':
        trainer = joint_trainer(args, framework, device)
        test_loss_keypoint, test_acc_class = trainer.test(test_loader)
        print("> Test on meta model,  task : {}".format(args.task))
        if args.task=="classification":
            print("test acc={:.4f}".format(test_acc_class))
        elif args.task=="keypoint":
            print("train loss={:.4f}".format(test_loss_keypoint))
        print("done Meta model testing")

    elif args.mode=='baseline_train':
        trainer = baseline_trainer(args, framework, device)
        trainer.train(train_loader, test_loader)
        print("done Baseline training")
    
    elif args.mode=='baseline_test':
        trainer = baseline_trainer(args, framework, device)
        test_acc_class = trainer.test(test_loader)
        print("> Test on baseline,  task : {}".format(args.task))
        if args.task=="classification":
            print("test acc={:.4f}".format(test_acc_class))
        elif args.task=="keypoint":
            print("baseline only classification task")
        print("done Baseline testing")
        
    else:
        print("Wrong mode...")
    