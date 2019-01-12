from __future__ import division
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np
from collections import OrderedDict


def conv3(in_channels, out_channels, stride=1):
    """ 3x1 convolutional layer for Bottleneck """
    return nn.Conv2d(in_channels, out_channels, kernel_size=(3,1), padding=(1,0), bias=False, stride=stride)

def conv1(in_channels, out_channels, stride=1):
    """ 1x1 convolutional layer for Bottleneck """
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False, stride=stride)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        Implementation of the Bottleneck block from the ResNeXt architecture.

        Args:
            in_channels: size of input channels
            out_channels: size of output channels
            stride: stride for the convolutional layers
            downsample: method for increasing channels width of input
        """
        super(Bottleneck, self).__init__()
        self.conv1 = conv1(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv1(out_channels, out_channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
 
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, layers, num_classes, block=Bottleneck):
        super(ResNet, self).__init__()
        self.init_planes = 64
        self.num_classes = num_classes
        self.init_conv = nn.Conv2d(3, self.init_planes, kernel_size=(7,1), 
                                stride=(1,1), padding=(3,0), bias=False)
        self.init_bn = nn.BatchNorm2d(self.init_planes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=(2,1), stride=(1,1))
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.layer3 = self._make_layer(block, 256, layers[2])
        self.layer4 = self._make_layer(block, 512, layers[3])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1,5))
        self.adaptive_pool_1x1 = nn.AdaptiveAvgPool2d((1,1))

    def _make_layer(self, block, out_channels, blocks):
        downsample = None
        if self.init_planes != out_channels * block.expansion:
            downsample = nn.Sequential(conv1(self.init_planes, out_channels * block.expansion),
                                        nn.BatchNorm2d(out_channels * block.expansion))
        layers = []
        layers.append(block(self.init_planes, out_channels, 1, downsample))
        self.init_planes = out_channels * block.expansion
        for _ in range(1,blocks):
            layers.append(block(self.init_planes, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        result = self.init_conv(x)
        result = self.init_bn(result)
        result = self.relu(result)
        #result = self.avgpool(result)
        result = self.layer1(result)
        result = self.avgpool(result)
        result = self.layer2(result)
        result = self.layer3(result)
        result = self.layer4(result)
        result = self.adaptive_pool(result)
        result = torch.transpose(result, 1, 3)
        result = self.adaptive_pool_1x1(result)
        result = result.view(-1, self.num_classes)
        result = F.softmax(result, dim=1)
        return result
    


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
        out = super(_DenseLayer, self).forward(x)
        return torch.cat([x, out],1)

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
                layer_config=(6,12,24,16), init_layer=64, expansion=4, reduce=False):
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
            block = _DenseBlock(layer_count, channels, growth_rate=growth_rate, expansion=expansion)
            channels = channels + layer_count * growth_rate
            self.features.add_module('denseblock%d' % (i+1), block)
            
            #Transitions
            if i != len(layer_config) - 1 and reduce:
                trans = _Transition(channels, channels // 2)
                self.features.add_module('transition%d' % (i+1), trans)
                channels = channels // 2
        
        #Decision layer
        self.features.add_module('final_pool', nn.AdaptiveAvgPool2d((1,num_classes)))
        self.features.add_module('final_relu', nn.ReLU(inplace=True))
        self.final_pool = nn.AdaptiveAvgPool2d((1,1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        result = self.features(x)
        result = torch.transpose(result,1,3)
        result = self.final_pool(result)
        result = result.view(-1,self.num_classes)
        result = F.softmax(result, dim=1)
        return result

# To be implemented
# class ResNeXt(nn.Module):
#     def __init__(self, bottleneck, depth, cardinality, base_width, num_classes):
#         super(ResNeXt, self).__init__()
#         assert (depth == 50 or depth == 101 or depth == 152), 'Depth value wrong!'

#         self.cardinality = cardinality
#         self.base_width = base_width
#         self.num_classes = num_classes
#         self.expansion = 128

#         self.initial_conv = nn.Conv2d(3,self.expansion,(3,1),1,(1,0),bias=True)
#         self.initial_bn = nn.BatchNorm2d(self.expansion)

#         self.in_channels = self.expansion
#         self.stage_1 = self._make_layer(bottleneck)