import torch.nn as nn

class VGGNet(nn.Module):
    def __init__(self, args):
        super(VGGNet, self).__init__()

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
        self.conv_1 = nn.Sequential(
            # 1
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1,1)),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1),
            self.activation,
            # 2
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1),
            self.activation,
            # pool
            nn.MaxPool2d((1,2)),
            self.dropout
        )
        self.conv_2 = nn.Sequential(
            # 1
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1),
            self.activation,
            # 2
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1),
            self.activation,
            # pool
            nn.MaxPool2d((1, 2)),
            self.dropout
        )
        self.conv_3 = nn.Sequential(
            # 1
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1),
            self.activation,
            # 2
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1),
            self.activation,
            # 3
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1),
            self.activation,
            # pool
            nn.MaxPool2d((1, 2)),
            self.dropout
        )
        self.conv_4 = nn.Sequential(
            # 1
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1),
            self.activation,
            # 2
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1),
            self.activation,
            # 3
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1),
            self.activation,
            # pool
            nn.MaxPool2d((1, 2)),
            self.dropout
        )

        """linear classifier"""
        self.linear = nn.Linear(in_features=11776, out_features=2, bias=True)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = out.view(x.shape[0],-1)
        out = self.linear(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)



            # if isinstance(m, nn.Conv2d):
            #     nn.init.kaiming_normal_(m.weight.data)
            #     if m.bias is not None:
            #         m.bias.data.zero_()
            # elif isinstance(m, nn.BatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     nn.init.normal_(m.weight.data, 0, 0.01)
            #     if m.bias is not None:
            #         m.bias.data.zero_()