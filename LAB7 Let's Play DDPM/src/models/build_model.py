import os
import torch
from models.unet import Unet


def build_model(args, n_classes, device):
    if args.mode=='train':
        print("\nBuilding models for training...")
        model = Unet(in_channels=args.in_channels, n_feature=args.n_feature, n_classes=n_classes)
        model.initialize_weights()
        return model.to(device)
    elif args.mode=='test':
        print("\nBuilding models for testing...")
        model_test = Unet(in_channels=args.in_channels, n_feature=args.n_feature, n_classes=n_classes)
        model_test.initialize_weights()
        model_test_new = Unet(in_channels=args.in_channels, n_feature=args.n_feature, n_classes=n_classes)
        model_test_new.initialize_weights()
        return model_test.to(device), model_test_new.to(device)