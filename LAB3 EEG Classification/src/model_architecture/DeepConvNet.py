import torch.nn as nn

class DeepConvNet(nn.Module):
    def __init__(self, args):
        super(DeepConvNet, self).__init__()

        """set activation function layer based on args.activation_type"""
        if args.activation_type == 'relu':
            self.activation = nn.ReLU()
        elif args.activation_type == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif args.activation_type == 'elu':
            self.activation = nn.ELU()

        """set dropout"""
        self.dropout = nn.Dropout(0.5)

        """conv modules"""
        self.conv = nn.Conv2d(1, 25, kernel_size=(1, 5), padding="valid")
        self.conv1 = nn.Sequential(
            nn.Conv2d(25, 25, kernel_size=(2, 1), padding="valid"),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1),
            self.activation,
            nn.MaxPool2d((1,2)),
            self.dropout
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 5), padding="valid"),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.1),
            self.activation,
            nn.MaxPool2d((1, 2)),
            self.dropout
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5), padding="valid"),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1),
            self.activation,
            nn.MaxPool2d((1, 2)),
            self.dropout
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5), padding="valid"),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1),
            self.activation,
            nn.MaxPool2d((1, 2)),
            self.dropout
        )
        """linear classifier"""
        self.linear = nn.Linear(in_features=8600, out_features=2, bias=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.view(x.shape[0],-1)
        out = self.linear(out)
        return out

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