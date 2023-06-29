import argparse
import numpy as np
import os
import random
import torch
from torchvision.utils import save_image

# self-define import
from data.dataloader import load_train_data, load_test_data
from models.build_model import build_model
from pipeline.ddpm import DDPM
from pipeline.trainer import trainer




def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train",type=str, choices=["train", "test"], 
                        help="train or test model")
    # hyper-parameters
    parser.add_argument("--seed", default=1, type=int, help="manual seed")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("--n_epoch", default=500, type=int, help="# training epochs")

    # Diffusion
    parser.add_argument('--beta_start', default=1e-4, type=float, help='start beta value')
    parser.add_argument('--beta_end', default=0.02, type=float, help='end beta value')
    parser.add_argument('--noise_steps', default=1000, type=int, help='frequency of sampling')
    # Training
    parser.add_argument("--img_size", default=64, type=int, help='image size')
    parser.add_argument("--in_channels", default=3, type=int, help="# channels of input images")
    parser.add_argument('--n_feature', default=256, type=int, 
                        help='time/condition embedding and feature maps dimension')
    parser.add_argument('--num_workers', default=2, type=int, help='workers of Dataloader')
    parser.add_argument('--noise_type', default='linear', type=str, 
                        choices=['linear','cosine'], help='noise schedule type')
    parser.add_argument('--resume', default=False, help='resume training')
    # parser.add_argument("--lr_scheduler", default="linear_decay", type=str, choices=["linear", "cosine"]")
    # path
    parser.add_argument("--dataset_path", default="/home/lin/Desktop/hw7_new/dataset", type=str, help="root of dataset dir")
    # parser.add_argument("--model_path", default="/home/lin/Desktop/hw7_new/ckpt_downcond/", type=str, help="model ckpt path")
    parser.add_argument("--model_path", default="/home/lin/Desktop/hw7_new/ckpt_downcond_cont/", type=str, help="model ckpt path")
    # parser.add_argument("--result_path", default="/home/lin/Desktop/hw7_new/result_downcond/", type=str, help="save img path")
    parser.add_argument("--result_path", default="/home/lin/Desktop/hw7_new/result_downcond_cont/", type=str, help="save img path")
    args = parser.parse_args()
    return args    


if __name__ == "__main__":

    """check cuda"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:{}'.format(device))

    """get param"""
    args = arg_parse()
    if  args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    """train or test"""
    if args.mode=='train':
        train_loader, test_loader, test_new_loader, n_classes = load_train_data(args, device)
        train_model = build_model(args, n_classes ,device)
        train_ddpm = DDPM(unet_model=train_model, betas=(args.beta_start, args.beta_end), 
                          noise_steps=args.noise_steps, device=device).to(device)
        if args.resume==True:
            print("Resume training...")
            path = os.path.join("/home/lin/Desktop/hw7_new/ckpt_downcond/", args.noise_type, "model_test_new.pth")
            train_ddpm.load_state_dict(torch.load(path))
            # args.lr = 1e-6
            # args.lr = 7.2892e-07
        trainer = trainer(args, train_ddpm, device)
        print("Start training...")
        trainer.train(train_loader, test_loader, test_new_loader)  
        print()


    elif args.mode=='test':
        test_loader, test_new_loader, n_classes = load_test_data(args, device)
        test_model, test_new_model = build_model(args, n_classes ,device)

        # load test ddpm model
        test_ddpm = DDPM(unet_model=test_model, betas=(args.beta_start, args.beta_end), 
                          noise_steps=args.noise_steps, device=device).to(device)
        path = os.path.join(args.model_path, args.noise_type, "model_test.pth")
        test_ddpm.load_state_dict(torch.load(path))
        trainer_test = trainer(args, test_ddpm, device)
        test_score, grid_test = trainer_test.test(test_loader)
        # save test img
        path = os.path.join(args.result_path, "test_{}.png".format('best'))
        save_image(grid_test, path)

        # load test new ddpm model
        test_new_ddpm = DDPM(unet_model=test_new_model, betas=(args.beta_start, args.beta_end),
                             noise_steps=args.noise_steps, device=device).to(device)
        path = os.path.join(args.model_path, args.noise_type, "model_test_new.pth")
        test_new_ddpm.load_state_dict(torch.load(path))
        trainer_test_new = trainer(args, test_new_ddpm, device)
        test_new_score, grid_test_new = trainer_test_new.test(test_new_loader)
        # save test new img
        path = os.path.join(args.result_path, "test_new_{}.png".format('best'))
        save_image(grid_test_new, path)

        print("test acc: {:.4f}, new test acc:{:.4f}".format(test_score, test_new_score))
        print("test done")
