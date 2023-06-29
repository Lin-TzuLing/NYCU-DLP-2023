import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights


class resnet(nn.Module):
    def __init__(self, model_type, pretrain_flag, num_classes):
        super(resnet, self).__init__()

        """ define model"""
        if model_type=="resnet18":
            if pretrain_flag==True:
                self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
                self.set_param_requires_grad(self.resnet, True)

            else:
                self.resnet = resnet18(weights=None)
        elif model_type=="resnet50":
            if pretrain_flag==True:
                self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
                self.set_param_requires_grad(self.resnet, True)
            else:
                self.resnet = resnet50(weights=None)

        """reinitialize last layer of model (output_dim = num_classes of this task)"""
        self.in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features=self.in_features, out_features=50)

        """self-define layers"""
        self.fc2 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=50, out_features=num_classes)
        )

    def forward(self, x):
        out = self.resnet(x)
        out = self.fc2(out)
        return out

    def set_param_requires_grad(self, model, feature_extract):
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False
        else:
            for param in model.parameters():
                param.requires_grad = True

    def initialize_weights(self):
        nn.init.normal_(self.resnet.fc.weight.data, 0, 0.01)
        self.resnet.fc.bias.data.zero_()
        nn.init.normal_(self.fc2[2].weight.data, 0, 0.01)
        self.fc2[2].bias.data.zero_()