from model_architecture.resnet import resnet


def build_model(args, num_classes):
    model = resnet(args.model_type, args.pretrain_flag, num_classes)
    return model