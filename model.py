from __future__ import division
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from collections import OrderedDict

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride, cardinality, base_width):
        """
        Implementation of the Bottleneck block from the ResNeXt architecture.

        in_channels: size of input channels
        out_channels: size of output channels
        stride: stride for the convolutional layers
        cardinality: cardinality as described in the paper
        base_width:
        """
        super(Bottleneck, self).__init__()
        D = int(math.floor(out_channels * (base_width / 64.)))
        C = cardinality

        self.conv_reduce = nn.Conv2d(in_channels, D*C, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D*C)

        self.conv_group = nn.Conv2d(D*C, D*C, kernel_size=(3,1), stride=1, padding=(1,0), groups=C, bias=False)
        self.bn_group = nn.BatchNorm2d(D*C)

        self.conv_expand = nn.Conv2d(D*C, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        residual = x

        out = self.conv_reduce(x)
        out = F.relu(self.bn_reduce(x), inplace=True)

        out = self.conv_group(x)
        out = F.relu(self.bn_group(x), inplace=True)
        
        out = self.conv_expand(x)
        out = F.relu(self.bn_expand(x) + residual, inplace=True)

        return out

class _DenseLayer(nn.Sequential):
    def __init__(self, input_features, growth_rate, expansion):
        """ 
        This is an implementation of Dense layers from the DenseNet
        convolutional architecture.

        input_features: size of input channels
        growth_rate: base number of layers
        expansion: amount of channel expansion
        """
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(input_features, expansion *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(expansion * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(expansion * growth_rate, growth_rate,
                        kernel_size=(3,1), stride=1, padding=(1,0), bias=False))

    def forward(self, x):
        return torch.cat([x, super().forward(x)],1)

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, input_features, growth_rate, expansion):
        """ Convolutional block that consists of DenseLayer layers."""
        super(_DenseBlock, self).__init__()
        for n in range(num_layers):
            layer = _DenseLayer(input_features + (n * growth_rate), growth_rate, expansion)
            self.add_module('dense_layer%d' % (n+1), layer)

class _Transition(nn.Sequential):
    def __init__(self, input_features, output_features):
        super(_Transition, self).__init__()

        self.add_module('norm', nn.BatchNorm2d(input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(input_features, output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=(2,1), stride=(2,1)))

class DenseNet(nn.Module):
    def __init__(self, num_classes, growth_rate=32, 
                layer_config=(6,12,24,48), init_layer=64, expansion=4):
        super(DenseNet, self).__init__()
        self.num_classes = num_classes

        #Initial layer
        self.features = nn.Sequential(OrderedDict([
            ('init_conv', nn.Conv2d(3, init_layer, kernel_size=1, stride=1, bias=False)),
            ('init_bn', nn.BatchNorm2d(init_layer)),
            ('init_relu', nn.ReLU(inplace=True))
        ]))

        #Dense blocks
        channels = init_layer
        for i, layer_count in enumerate(layer_config):
            block = _DenseBlock(layer_count, channels, growth_rate, expansion)
            channels += layer_count * expansion
            self.features.add_module('denseblock%d' % (i+1), block)
            
            #Transitions
            if i != len(layer_config) - 1:
                trans = _Transition(channels, channels // 2)
                self.features.add_module('transition%d' % (i+1), trans)
                channels = channels // 2
        
        #Decision layer
        self.features.add_module('final_pool', nn.AdaptiveAvgPool2d((1,num_classes)))
        self.features.add_module('final_conv', nn.Conv2d(channels, num_classes, (1,num_classes), bias=False))
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1,self.num_classes)
        x = F.softmax(x)
        return x
        
# To be implemented
# class ResNeXt(nn.Module):
#     def __init__(self, bottleneck, depth, cardinality, base_width, num_classes):
#         super(ResNeXt, self).__init__()
#         assert (depth == 50 or depth == 101 or depth == 152), 'Depth value wrong!'

#         self.cardinality = cardinality
#         self.base_width = base_width
#         self.num_classes = num_classes
#         self.expansion = 128

#         self.initial_conv = nn.Conv2d(3,self.expansion,(3,1),1,(1,0),bias=False)
#         self.initial_bn = nn.BatchNorm2d(self.expansion)

#         self.in_channels = self.expansion
#         self.stage_1 = self._make_layer(bottleneck)
