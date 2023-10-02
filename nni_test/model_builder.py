from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   groups=in_channels, bias=bias, padding=(kernel_size - 1) // 2)
        self.pointwise = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.pointwise(self.depthwise(x))
        return out




def build_model(args):
    input_dim = (1, 3, 32, 32)
    output_size = 10

    model = nn.Sequential()

    conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
    conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)

    if (args['conv2'] == 'normal'):
        conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
    elif (args['conv2'] == 'depthwise'):
        conv2 = DepthwiseSeparableConv2d(32, 64)

    dropout = nn.Dropout(args['dropout_rate'])

    model.add_module('conv1', conv1)
    model.add_module('relu1', nn.ReLU())
    model.add_module('conv2', conv2)
    model.add_module('relu2', nn.ReLU())
    model.add_module('dropout', dropout)
    model.add_module('flatten', nn.Flatten(1))

    conv_out_size = np.prod(
        model(torch.zeros((1, *input_dim[1:]))).shape
    )

    print(conv_out_size)

    fc1 = nn.Linear(conv_out_size, args['fc1'])
    fc2 = nn.Linear(args['fc1'], output_size)

    print(fc2(fc1(torch.zeros(1, conv_out_size))))

    model.add_module('fc1', fc1)
    model.add_module('relu3', nn.ReLU())
    model.add_module('fc2', fc2)

    lr = args.setdefault("lr", 1e-3)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, optimizer

