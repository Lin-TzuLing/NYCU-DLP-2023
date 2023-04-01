import numpy as np

class fc_layer:
    """fully connected layer module"""
    def __init__(self, in_dim, out_dim, data_type, activation_type):
        # param
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.data_type = data_type
        self.activation_type = activation_type

        # weight, bias (initial normal distribution, none zero)
        self.weight = np.random.randn(self.in_dim, self.out_dim)
        self.bias = np.random.randn(1, self.out_dim)

        # gaussian random numbers with zero mean and 0.01/0.1 standard deviations
        if self.activation_type=='none' and self.data_type=='linear':
            self.weight *= 0.01
            self.bias *= 0.01
        if self.activation_type=='relu' and self.data_type=='xor':
            # He initialization
            self.weight *= 1 / np.sqrt(self.in_dim/2)
            self.weight *= 1 / np.sqrt(self.in_dim/2)
            # self.weight = self.weight * 0.1
            # self.bias = self.bias * 0.1

        # gradient (for backward propagation)
        self.grad_weight = np.zeros((self.in_dim, self.out_dim))
        self.grad_bias = np.zeros((1, self.out_dim))

    def forward(self, x):
        # save input for backward propagation
        self.input = x
        out = np.dot(x,self.weight)+self.bias
        return out

    def backward(self, grad):
        # need to be same shape of weight matrix -> input.transpose()
        self.grad_weight = np.dot(self.input.transpose(), grad)
        # row sum, same grad, delta(z)/delta(b)=1 from z=wx+b
        self.grad_bias = grad.sum(axis=0, keepdims=True)
        # grad as previous node, dot weight to propagate back (shape of grad)
        out = np.dot(grad, self.weight.transpose())
        return out

    def update(self, lr):
        self.weight -= lr * self.grad_weight
        self.bias -= lr * self.grad_bias


class activation:
    """activation function module"""
    def __init__(self, activation_type):
        self.activation_type = activation_type

    def forward(self,x):
        if self.activation_type=='sigmoid':
            # backward propagation of activation: derivative(a), save 'a' for backward.
            self.act_out = 1.0/(1.0+np.exp(-x))
            return self.act_out
        elif self.activation_type=='none':
            # self.act_out = x
            return x
        elif self.activation_type=='relu':
            self.act_input = x
            return np.maximum(x, 0)
        else:
            raise ValueError('incorrect activation type')

    def backward(self,grad):
        if self.activation_type=="sigmoid":
            activation_derivative = np.multiply(self.act_out, 1.0 - self.act_out)
            # activation_derivative = constant, use multiply
            return activation_derivative * grad
        elif self.activation_type=='none':
            return grad
        elif self.activation_type=='relu':
            grad_out = grad.copy()
            # position of original input <=0, its gradient = 0
            grad_out[self.act_input <= 0] = 0
            return grad_out
        else:
            raise ValueError('incorrect activation type')

