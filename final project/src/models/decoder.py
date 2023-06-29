import torch.nn as nn



class decoder(nn.Module):
    def __init__(self, model_type, n_keypoint_coord):
        super(decoder, self).__init__()

        """ define model"""
        if model_type=="resnet18":
            self.decoder = nn.Sequential(
                                        nn.Linear(512, 256), 
                                        #  nn.Linear(512//2, 256//2), 
                                        nn.ReLU(),
                                        nn.Linear(256, n_keypoint_coord),
                                        # nn.Linear(256//2, n_keypoint_coord),
                                        nn.Sigmoid()
                                        )

        elif model_type=="resnet50":
            self.decoder = nn.Sequential(
                                        nn.Linear(2048, 1024),
                                        # nn.Linear(2048//2, 1024//2),
                                        nn.ReLU(),
                                        nn.Linear(1024, 512),
                                        nn.Dropout(0.2),
                                        nn.ReLU(),
                                        nn.Linear(512, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, n_keypoint_coord),
                                        # nn.Linear(1024//2, n_keypoint_coord),
                                        nn.Sigmoid()
                                        )
        
    def forward(self, x):
        out = self.decoder(x)
        return out

    def initialize_weights(self):
        for m in self.modules():  
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()