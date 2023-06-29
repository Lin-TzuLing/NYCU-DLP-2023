import torch.nn as nn



class classifier(nn.Module):
    def __init__(self, model_type, n_classes):
        super(classifier, self).__init__()

        """ define model"""
        if model_type=="resnet18":      
            # self.clssifier = nn.Sequential(
            #                                 nn.Linear(512//2, 256//2),
            #                                 nn.Linear(256//2, n_classes)
            #                                 )
            # self.clssifier = nn.Sequential(
            #                                 nn.Linear(512, 256),
            #                                 nn.Linear(256, n_classes)
            #                                 )
            self.clssifier = nn.Linear(512, n_classes)
        elif model_type=="resnet50":
            # self.clssifier = nn.Sequential(
            #                                 nn.Linear(2048//2, 1024//2),
            #                                 nn.Linear(1024//2, n_classes)
            #                                 )
            self.clssifier = nn.Sequential(
                                            nn.Linear(2048, 1024),
                                            nn.Linear(1024, 512),
                                            nn.Linear(512, n_classes)
                                            )
            # self.clssifier = nn.Linear(2048//2, n_classes)   
            # self.clssifier = nn.Linear(2048, n_classes)     

    def forward(self, x):
        out = self.clssifier(x)
        return out

    def initialize_weights(self):
        for m in self.modules():  
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()