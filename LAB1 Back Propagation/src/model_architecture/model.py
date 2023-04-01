# self import
from model_architecture.layer import fc_layer, activation

class NNmodel:
    """build model_architecture"""
    def __init__(self,in_dim, hidden_dim, out_dim, data_type, activation_type):
        self.fc1 = fc_layer(in_dim=in_dim, out_dim=hidden_dim,
                            data_type=data_type, activation_type=activation_type)
        self.fc2 = fc_layer(in_dim=hidden_dim, out_dim=hidden_dim,
                            data_type=data_type, activation_type=activation_type)
        self.fc3 = fc_layer(in_dim=hidden_dim, out_dim=out_dim,
                            data_type=data_type, activation_type=activation_type)
        self.act= activation(activation_type=activation_type)
        # output activation must not be none
        if activation_type=='none':
            self.out_act = activation(activation_type='sigmoid')
        else:
            self.out_act = activation(activation_type=activation_type)

    def forward(self, x):
        """forward propagation"""
        self.z1 = self.fc1.forward(x)
        self.a1 = self.act.forward(self.z1)
        self.z2 = self.fc2.forward(self.a1)
        self.a2 = self.act.forward(self.z2)
        self.z3 = self.fc3.forward(self.a2)
        self.out = self.out_act.forward(self.z3)
        return self.out

    def backward(self, grad_y):
        """backward from last layer to first layer"""
        grad_out = self.out_act.backward(grad_y)
        grad_z3 = self.fc3.backward(grad_out)
        grad_a2 = self.act.backward(grad_z3)
        grad_z2 = self.fc2.backward(grad_a2)
        grad_a1 = self.act.backward(grad_z2)
        grad_z1 = self.fc1.backward(grad_a1)

    def update(self, lr):
        """update each fc layer"""
        self.fc1.update(lr)
        self.fc2.update(lr)
        self.fc3.update(lr)