#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 20:58:34 2018
@author: harry
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.autograd import Variable
from utils import LearnedGroupConv1D

def _bn_function_factory(conv, norm, prelu):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = norm(prelu(conv(concated_features)))
        return bottleneck_output

    return bn_function

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate):
        super(_DenseLayer, self).__init__()
        
        self.conv0 = LearnedGroupConv1D(num_input_features, 4 * growth_rate, kernel_size=3, padding=1, groups=4, condense_factor=4)
        self.norm0 = nn.BatchNorm1d(4 * growth_rate)

        self.PReLU = nn.PReLU(num_parameters=1, init=0.05)
        
        self.conv1 = LearnedGroupConv1D(4 * growth_rate, growth_rate, kernel_size=3, padding=1, groups=8, condense_factor=8)
        self.norm1 = nn.BatchNorm1d(growth_rate)

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.conv0, self.norm0, self.PReLU)
        if any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.norm1(self.conv1(bottleneck_output))
        return new_features
    
class _DenseBlock(nn.Module):
    def __init__(self, nChannels, growth_rate):
        super(_DenseBlock, self).__init__()
        layer = _DenseLayer(nChannels, growth_rate)
        self.add_module('denselayer', layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class EdgeRNNG(nn.Module):
    def __init__(self, num_class):
        super(EdgeRNNG, self).__init__()

        self.output_size = num_class
        num_features = 152
        growth_rate = 16

        self.dense1 = _DenseBlock(num_features,growth_rate)
        num_features = num_features + growth_rate
        self.dense2 = _DenseBlock(num_features,growth_rate)
        num_features = num_features + growth_rate
        self.dense3 = _DenseBlock(num_features,growth_rate)
        num_features = num_features + growth_rate
        self.dense4 = _DenseBlock(num_features,growth_rate)
        num_features = num_features + growth_rate
        self.dense5 = _DenseBlock(num_features,growth_rate)
        num_features = num_features + growth_rate
        self.dense6 = _DenseBlock(num_features,growth_rate)
        num_features = num_features + growth_rate
        self.dense7 = _DenseBlock(num_features,growth_rate)
        num_features = num_features + growth_rate
        self.dense8 = _DenseBlock(num_features,growth_rate)
        num_features = num_features + growth_rate
        self.dense9 = _DenseBlock(num_features,growth_rate)
        num_features = num_features + growth_rate
        self.dense10 = _DenseBlock(num_features,growth_rate)
        num_features = num_features + growth_rate
        self.dense11 = _DenseBlock(num_features,growth_rate)
        num_features = num_features + growth_rate
        self.dense12 = _DenseBlock(num_features,growth_rate)
        num_features = num_features + growth_rate
        
        self.MaxPool1d = nn.MaxPool1d(2,2)
        
        self.RNN = nn.RNN(num_features, num_features, 1, batch_first=True,
                              dropout=0,bidirectional=False)
        self.Tanh = nn.Tanh()
        self.out = nn.Linear(num_features, self.output_size)

    
    def forward(self, x):
        
        x = x.float()

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.dense6(x)
        x = self.dense7(x)
        x = self.dense8(x)
        x = self.dense9(x)
        x = self.dense10(x)
        x = self.dense11(x)
        x = self.dense12(x)
        
        x = self.MaxPool1d(x)
        
        x = x.permute(0, 2, 1)
        x, h_n = self.RNN(x)
        
        residual = x 
        x = F.sigmoid(x)
        residual = residual * x
        residual = (residual.sum(1)) / (x.sum(1))
        residual = self.Tanh(residual)

        x = self.out(residual)
        
        return x
