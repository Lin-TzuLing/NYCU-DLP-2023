import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, args):
        super(EEGNet, self).__init__()

        """set activation function layer based on args.activation_type"""
        if args.activation_type == 'relu':
            self.activation = nn.ReLU()
        elif args.activation_type == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif args.activation_type == 'elu':
            self.activation = nn.ELU()

        """set dropout"""
        self.dropout = nn.Dropout(args.dropout_ratio)

        """conv modules"""
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1,51), padding=(0,25), bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.depthwiseConv = nn.Sequential(
            # group將in_channel分成16組，每組重複用(out_channel 32/groups 16)次
            nn.Conv2d(16, 32, kernel_size=(2,1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            self.activation,
            nn.AvgPool2d(kernel_size=(1,4), stride=(1,4), padding=0),
            self.dropout
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1,15), padding=(0,7), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            self.activation,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            self.dropout
        )

        """linear classifier"""
        self.linear = nn.Linear(in_features=736, out_features=2, bias=True)

    def forward(self, x):
        out = self.firstconv(x)
        out = self.depthwiseConv(out)
        out = self.separableConv(out)
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