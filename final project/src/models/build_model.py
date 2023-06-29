import torch.nn as nn
from models.resnet import resnet
from models.decoder import decoder
from models.classifier import classifier

def build_model(args, n_classes, n_keypoint_coord, device):
    model_type = args.arch
    model = resnet(model_type=model_type, pretrain_flag=args.pretrained)
    auxiliary_decoder = decoder(model_type=model_type, n_keypoint_coord=n_keypoint_coord)
    downstream_classifier = classifier(model_type=model_type, n_classes=n_classes)

    if args.pretrained==False:
        model.initialize_weights()
    auxiliary_decoder.initialize_weights()
    downstream_classifier.initialize_weights()

    framework = {
        "model": model.to(device),
        "auxiliary_decoder": auxiliary_decoder.to(device),
        "downstream_classifier": downstream_classifier.to(device), 
    }

    return framework