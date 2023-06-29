import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights


class resnet(nn.Module):
    def __init__(self, model_type, pretrain_flag):
        super(resnet, self).__init__()

        """ define model"""
        if model_type=="resnet18":
            if pretrain_flag==True:
                self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
                # self.set_param_requires_grad(self.resnet, True)
            else:
                self.resnet = resnet18(weights=None)

        elif model_type=="resnet50":
            if pretrain_flag==True:
                self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
                # self.set_param_requires_grad(self.resnet, True)
            else:
                self.resnet = resnet50(weights=None)


        # """reinitialize last layer of model (output_dim = num_classes of this task)"""
        self.in_features = self.resnet.fc.in_features
        # self.resnet.fc = nn.Linear(in_features=self.in_features, out_features=50)
        """ignore last fc in resnet model"""
        self.resnet.fc = nn.Identity()
        self.conv = nn.Conv2d(in_channels=self.in_features, 
                              out_channels=self.in_features//2, 
                              kernel_size=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.resnet(x)
        # out = out.unsqueeze(-1).unsqueeze(-1)
        # out = self.relu(self.conv(out)).squeeze()
        return out

    def set_param_requires_grad(self, model, feature_extract):
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False
        else:
            for param in model.parameters():
                param.requires_grad = True

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
        # nn.init.normal_(self.resnet.fc.weight.data, 0, 0.01)
        # self.resnet.fc.bias.data.zero_()
        # nn.init.normal_(self.fc2[2].weight.data, 0, 0.01)
        # self.fc2[2].bias.data.zero_()