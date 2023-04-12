# self-import
from model_architecture.EEGNet import EEGNet
from model_architecture.DeepConvNet import DeepConvNet
from model_architecture.VGGNet import VGGNet

def build_model(args):
    if args.model_type == "eeg":
        model = EEGNet(args)
    elif args.model_type == 'deepconv':
        model = DeepConvNet(args)
    elif args.model_type == 'vgg':
        model = VGGNet(args)
    else:
        raise ValueError("incorrect model type")

    return model