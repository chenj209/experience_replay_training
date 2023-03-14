import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import numpy as np

# the input dimension is modified to 2 # block
class skip_layer(nn.Module):
    def __init__(self, m, activation):
        super(skip_layer, self).__init__()
        self.fc1 = nn.Linear(m, m)
        self.fc2 = nn.Linear(m, m)
        if activation=='relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation== 'tanh':
            self.activation = nn.Tanh()
        elif activation== 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("should choose a valid activation function")

    def forward(self, x):
        s = x # 2
        y = self.fc1(x) # 3
        y = self.activation(y)
        y = self.fc2(y) # 4
        y = self.activation(y)
        y = y + s
        return y

class ResNet_output30(nn.Module):
    def __init__(self, m, activation, num_blocks):
        super(ResNet_output30, self).__init__()
        self.fc1 = nn.Linear(122, m)
        self.input_size = 122
        self.fc2 = nn.Linear(m, m)

        self.basicblocks = self._make_layer(m, num_blocks-1, activation)

        self.outlayer = nn.Linear(m, 30, bias=True)
        if activation=='relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation== 'tanh':
            self.activation = nn.Tanh()
        elif activation== 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("should choose a valid activation function")

        self.node_size = m

        for idx, m in enumerate(self.modules()):
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_layer(self, m, n_layers, activation):
        layers = []
        for i in range(n_layers):
            layers.append(skip_layer(m, activation))
        return nn.Sequential(*layers)

    def forward(self, x):
        s = torch.nn.functional.pad(x,(0,self.node_size-self.input_size))
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        y = self.activation(y)
        y = y + s

        y = self.basicblocks(y)
        output = self.outlayer(y)
        return output

class ResNet_output1(nn.Module):
    def __init__(self, m, activation, num_blocks):
        super(ResNet_output1, self).__init__()
        self.fc1 = nn.Linear(122, m)
        self.input_size = 122
        self.fc2 = nn.Linear(m, m)

        self.basicblocks = self._make_layer(m, num_blocks-1, activation)

        self.outlayer = nn.Linear(m, 1, bias=True)
        if activation=='relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation== 'tanh':
            self.activation = nn.Tanh()
        elif activation== 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("should choose a valid activation function")

        self.node_size = m

        for idx, m in enumerate(self.modules()):
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_layer(self, m, n_layers, activation):
        layers = []
        for i in range(n_layers):
            layers.append(skip_layer(m, activation))
        return nn.Sequential(*layers)

    def forward(self, x):
        s = torch.nn.functional.pad(x, (0, self.node_size-self.input_size))
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        y = self.activation(y)
        y = y + s

        y = self.basicblocks(y)
        output = self.outlayer(y)
        return output

class ResNet_output4(nn.Module):
    def __init__(self, m, activation, num_blocks):
        super(ResNet_output4, self).__init__()
        self.fc1 = nn.Linear(122, m)
        self.input_size = 122
        self.fc2 = nn.Linear(m, m)

        self.basicblocks = self._make_layer(m, num_blocks-1, activation)

        self.outlayer = nn.Linear(m, 4, bias=True)
        if activation=='relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation== 'tanh':
            self.activation = nn.Tanh()
        elif activation== 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("should choose a valid activation function")

        self.node_size = m

        for idx, m in enumerate(self.modules()):
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_layer(self, m, n_layers, activation):
        layers = []
        for i in range(n_layers):
            layers.append(skip_layer(m, activation))
        return nn.Sequential(*layers)

    def forward(self, x):
        s = torch.nn.functional.pad(x,(0,self.node_size-self.input_size))
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        y = self.activation(y)
        y = y + s

        y = self.basicblocks(y)
        output = self.outlayer(y)
        return output

class ResNet_output5(nn.Module):
    def __init__(self, m, activation, num_blocks):
        super(ResNet_output5, self).__init__()
        self.fc1 = nn.Linear(122, m)
        self.input_size = 122
        self.fc2 = nn.Linear(m, m)

        # self.channel_max_y = torch.FloatTensor(norm_vec['channel_max_y'][0:1,61:66,0,0]).cuda()
        # self.channel_min_y = torch.FloatTensor(norm_vec['channel_min_y'][0:1,61:66,0,0]).cuda()
        # print('123', self.channel_min_y.size(), self.channel_min_y.size())
        # self.correction = (-self.channel_min_y-self.channel_max_y)/(self.channel_max_y-self.channel_min_y)
        self.basicblocks = self._make_layer(m, num_blocks-1, activation)

        self.outlayer = nn.Linear(m, 5, bias=True)
        if activation=='relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation== 'tanh':
            self.activation = nn.Tanh()
        elif activation== 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("should choose a valid activation function")

        self.node_size = m
        # self.relu = torch.nn.Softplus()#nn.ReLU(inplace=False)

        for idx, m in enumerate(self.modules()):
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_layer(self, m, n_layers, activation):
        layers = []
        for i in range(n_layers):
            layers.append(skip_layer(m, activation))
        return nn.Sequential(*layers)

    def forward(self, x):

        s = torch.nn.functional.pad(x,(0,self.node_size-self.input_size))
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        y = self.activation(y)
        y = y + s
        y = self.basicblocks(y)
        output = self.outlayer(y)
        # output = self.relu(output)
        # print('relu!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # output = self.tanh(output)+self.correction
        return output

    
class mlp_output1(nn.Module):
    def __init__(self, m, activation, num_blocks):
        super(mlp_output1, self).__init__()
        
        layers = []
        layers.append(nn.Linear(122, m))
        for i in range(num_blocks-2):
            if activation=='relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation== 'tanh':
                layers.append(nn.Tanh())
            elif activation== 'sigmoid':
                layers.append(nn.Sigmoid())
            else:
                raise ValueError("should choose a valid activation function")
            layers.append(nn.Linear(m, m))
        if activation=='relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation== 'tanh':
            layers.append(nn.Tanh())
        elif activation== 'sigmoid':
            layers.append(nn.Sigmoid())
        else:
            raise ValueError("should choose a valid activation function")
        layers.append(nn.Linear(m, 1, bias=True))

        self.nn = nn.Sequential(*layers)
        for idx, m in enumerate(self.modules()):
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward(self, x):
        output = self.nn(x)
        return output

class mlp_output4(nn.Module):
    def __init__(self, m, activation, num_blocks):
        super(mlp_output4, self).__init__()
        
        layers = []
        layers.append(nn.Linear(122, m))
        for i in range(num_blocks-2):
            if activation=='relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation== 'tanh':
                layers.append(nn.Tanh())
            elif activation== 'sigmoid':
                layers.append(nn.Sigmoid())
            else:
                raise ValueError("should choose a valid activation function")
            layers.append(nn.Linear(m, m))
        if activation=='relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation== 'tanh':
            layers.append(nn.Tanh())
        elif activation== 'sigmoid':
            layers.append(nn.Sigmoid())
        else:
            raise ValueError("should choose a valid activation function")
        layers.append(nn.Linear(m, 4, bias=True))

        self.nn = nn.Sequential(*layers)
        for idx, m in enumerate(self.modules()):
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward(self, x):
        output = self.nn(x)
        return output

class mlp_output30(nn.Module):
    def __init__(self, m, activation, num_blocks):
        super(mlp_output30, self).__init__()
        
        layers = []
        layers.append(nn.Linear(122, m))
        for i in range(num_blocks-2):
            if activation=='relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation== 'tanh':
                layers.append(nn.Tanh())
            elif activation== 'sigmoid':
                layers.append(nn.Sigmoid())
            else:
                raise ValueError("should choose a valid activation function")
            layers.append(nn.Linear(m, m))
        if activation=='relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation== 'tanh':
            layers.append(nn.Tanh())
        elif activation== 'sigmoid':
            layers.append(nn.Sigmoid())
        else:
            raise ValueError("should choose a valid activation function")
        layers.append(nn.Linear(m, 30, bias=True))

        self.nn = nn.Sequential(*layers)
        for idx, m in enumerate(self.modules()):
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward(self, x):
        output = self.nn(x)
        return output
    
class mlp_output5(nn.Module):
    def __init__(self, m, activation, num_blocks):
        super(mlp_output5, self).__init__()
        
        layers = []
        layers.append(nn.Linear(122, m))
        for i in range(num_blocks-2):
            if activation=='relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation== 'tanh':
                layers.append(nn.Tanh())
            elif activation== 'sigmoid':
                layers.append(nn.Sigmoid())
            else:
                raise ValueError("should choose a valid activation function")
            layers.append(nn.Linear(m, m))
        if activation=='relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation== 'tanh':
            layers.append(nn.Tanh())
        elif activation== 'sigmoid':
            layers.append(nn.Sigmoid())
        else:
            raise ValueError("should choose a valid activation function")
        layers.append(nn.Linear(m, 5, bias=True))

        self.nn = nn.Sequential(*layers)
        for idx, m in enumerate(self.modules()):
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward(self, x):
        output = self.nn(x)
        return output