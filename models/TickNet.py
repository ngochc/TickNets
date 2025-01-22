import torch.nn as nn
import torch
import re
import types

import torch.nn
import torch.nn.init


from .common import conv1x1_block, Classifier, conv3x3_dw_blockAll, conv3x3_block
from .SE_Attention import *


class FR_PDP_block(torch.nn.Module):
    """
    FR_PDP_block for TickNet.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride):
        super().__init__()
        self.Pw1 = conv1x1_block(in_channels=in_channels,
                                 out_channels=in_channels,
                                 use_bn=False,
                                 activation=None)
        self.Dw = conv3x3_dw_blockAll(channels=in_channels, stride=stride)
        self.Pw2 = conv1x1_block(in_channels=in_channels,
                                 out_channels=out_channels,
                                 groups=1)
        self.PwR = conv1x1_block(in_channels=in_channels,
                                 out_channels=out_channels,
                                 stride=stride)
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.SE = SE(out_channels, 16)

    def forward(self, x):
        residual = x
        x = self.Pw1(x)
        x = self.Dw(x)
        x = self.Pw2(x)
        x = self.SE(x)
        if self.stride == 1 and self.in_channels == self.out_channels:
            x = x + residual
        else:
            residual = self.PwR(residual)
            x = x + residual
        return x


class TickNet(nn.Module):
    """
    Class for constructing TickNet.
    """

    def __init__(self,
                 num_classes,
                 init_conv_channels,
                 init_conv_stride,
                 channels,
                 strides,
                 in_channels=3,
                 in_size=(224, 224),
                 use_data_batchnorm=True
                 ):

        super().__init__()
        self.use_data_batchnorm = use_data_batchnorm
        self.in_size = in_size

        self.backbone = torch.nn.Sequential()

       # data batchnorm
        if self.use_data_batchnorm:
            self.backbone.add_module("data_bn", torch.nn.BatchNorm2d(num_features=in_channels))

        # init conv
        self.backbone.add_module("init_conv", conv3x3_block(in_channels=in_channels, out_channels=init_conv_channels, stride=init_conv_stride))

        # stages
        in_channels = init_conv_channels
        in_channels = self.add_stages(in_channels, channels, strides)
        self.final_conv_channels = 1024        

        self.backbone.add_module("final_conv", conv1x1_block(in_channels=in_channels, out_channels=self.final_conv_channels, activation="relu"))
        self.backbone.add_module("global_pool", torch.nn.AdaptiveAvgPool2d(output_size=1))
        in_channels = self.final_conv_channels
        # classifier
        self.classifier = Classifier(in_channels=in_channels, num_classes=num_classes)

        self.init_params()

    def add_stages(self, in_channels, channels, strides):
        for stage_id, stage_channels in enumerate(channels):
            stage = torch.nn.Sequential()
            for unit_id, unit_channels in enumerate(stage_channels):
                stride = strides[stage_id] if unit_id == 0 else 1                
                stage.add_module("unit{}".format(unit_id + 1), FR_PDP_block(in_channels=in_channels, out_channels=unit_channels, stride=stride))
                in_channels = unit_channels
            self.backbone.add_module("stage{}".format(stage_id + 1), stage)
        return in_channels

    def init_params(self):
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.classifier.init_params()

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


class SpatialTickNet(TickNet):
    """
    Class for constructing SpatialTickNet, enhancing TickNet for spatial feature learning.
    """

    def __init__(self,
                 num_classes,
                 init_conv_channels,
                 init_conv_stride,
                 channels,
                 strides,
                 in_channels=3,
                 in_size=(224, 224),
                 use_data_batchnorm=True):
        super().__init__(num_classes, init_conv_channels, init_conv_stride,
                         channels, strides, in_channels, in_size, use_data_batchnorm)

    def add_stages(self, in_channels, channels, strides):
        for stage_id, stage_channels in enumerate(channels):
            stage = nn.Sequential()
            for unit_id, unit_channels in enumerate(stage_channels):
                stride = strides[stage_id] if unit_id == 0 else 1

                stage.add_module(
                    "unit{}".format(unit_id + 1),
                    FR_PDP_block(in_channels=in_channels,
                                 out_channels=unit_channels, stride=stride)
                )
                #print(f'add_stages: stage({stage_id + 1}), node({unit_id + 1}), stride({stride})')
                in_channels = unit_channels
            self.backbone.add_module("stage{}".format(stage_id + 1), stage)
        return in_channels

###
# %% model definitions
###


def build_TickNet(num_classes, typesize='small', cifar=False):
    init_conv_channels = 32
    channel_options = {
        'basic': [[128], [64], [128], [256], [512]],
        'small': [[128], [64, 128], [256, 512, 128], [64, 128, 256], [512]],
        'large': [[128], [64, 128], [256, 512, 128, 64, 128, 256], [512, 128, 64, 128, 256], [512]]
    }
    channels = channel_options.get(typesize, channel_options['small'])
    print(f'THE ACTUAL CHANNEL: {typesize}')
    if cifar:
        in_size = (32, 32)
        init_conv_stride = 1
        strides = [1, 1, 2, 2, 2]
    else:
        in_size = (224, 224)
        init_conv_stride = 2
        if typesize == 'basic':
            strides = [1, 2, 2, 2, 2]
        else:
            strides = [2, 1, 2, 2, 2]
    
    return TickNet(num_classes=num_classes,
                   init_conv_channels=init_conv_channels,
                   init_conv_stride=init_conv_stride,
                   channels=channels,
                   strides=strides,
                   in_size=in_size)

def get_cf(config_list, cf_index):
    """
    Safely get the configuration at cf_index from config_list.
    If cf_index is out of range, return the configuration at index 0.
    """
    if 0 <= cf_index < len(config_list):
        return config_list[cf_index]
    else:
        print(f"cf_index {cf_index} is out of range. Using default index 0.")
        return config_list[0]

def build_SpatialTickNet(num_classes, typesize='basic', cifar=False, cf_index=0):
    init_conv_channels = 32
    small_cf = [
        [[256], [128], [64], [128, 256, 512, 256, 128, 64, 128, 256], [512]], #r1
        [[256], [128], [64, 128], [256, 512, 256, 128, 64, 128, 256], [512]], #r1
        [[256], [128], [64, 128, 256], [512, 256, 128, 64, 128, 256], [512]], #r1
        [[256], [128], [64, 128, 256, 512], [256, 128, 64, 128, 256], [512]], #r1
        [[256], [128], [64, 128, 256, 512, 256], [128, 64, 128, 256], [512]], #r1
        [[256], [128], [64, 128, 256, 512, 256, 128], [64, 128, 256], [512]], #r1
        [[256], [128], [64, 128, 256, 512, 256, 128, 64], [128, 256], [512]], #r1
        [[256], [128], [64, 128, 256, 512, 256, 128, 64, 128], [256], [512]], #r1
        [[256], [128, 64], [128], [256, 512, 256, 128, 64, 128, 256], [512]], #r1
        [[256], [128, 64], [128, 256], [512, 256, 128, 64, 128, 256], [512]], #r1
        [[256], [128, 64], [128, 256, 512], [256, 128, 64, 128, 256], [512]], #r1
        [[256], [128, 64], [128, 256, 512, 256], [128, 64, 128, 256], [512]], #r1
        [[256], [128, 64], [128, 256, 512, 256, 128], [64, 128, 256], [512]], #r1
        [[256], [128, 64], [128, 256, 512, 256, 128, 64], [128, 256], [512]], #r1
        [[256], [128, 64], [128, 256, 512, 256, 128, 64, 128], [256], [512]], #r1
        [[256], [128, 64, 128], [256], [512, 256, 128, 64, 128, 256], [512]], #r1
        [[256], [128, 64, 128], [256, 512], [256, 128, 64, 128, 256], [512]], #r1
        [[256], [128, 64, 128], [256, 512, 256], [128, 64, 128, 256], [512]], # Current Best
        [[256], [128, 64, 128], [256, 512, 256, 128], [64, 128, 256], [512]], #r1
        [[256], [128, 64, 128], [256, 512, 256, 128, 64], [128, 256], [512]], #r1
        [[256], [128, 64, 128], [256, 512, 256, 128, 64, 128], [256], [512]], #r1
        [[256], [128, 64, 128, 256], [512], [256, 128, 64, 128, 256], [512]], #r1
        [[256], [128, 64, 128, 256], [512, 256], [128, 64, 128, 256], [512]], #r1
        [[256], [128, 64, 128, 256], [512, 256, 128], [64, 128, 256], [512]], #r1
        [[256], [128, 64, 128, 256], [512, 256, 128, 64], [128, 256], [512]], #r1
        [[256], [128, 64, 128, 256], [512, 256, 128, 64, 128], [256], [512]], #r1
        [[256], [128, 64, 128, 256, 512], [256], [128, 64, 128, 256], [512]], #r1
        #[[256], [128, 64, 128, 256, 512], [256, 128], [64, 128, 256], [512]], #r1 FLOP > 2
        #[[256], [128, 64, 128, 256, 512], [256, 128, 64], [128, 256], [512]], #r1 FLOP > 2
        #[[256], [128, 64, 128, 256, 512], [256, 128, 64, 128], [256], [512]], #r1 FLOP > 2
        #[[256], [128, 64, 128, 256, 512, 256], [128], [64, 128, 256], [512]], #r1 FLOP > 2
        #[[256], [128, 64, 128, 256, 512, 256], [128, 64], [128, 256], [512]], #r1 FLOP > 2
        #[[256], [128, 64, 128, 256, 512, 256], [128, 64, 128], [256], [512]], #r1 FLOP > 2
        #[[256], [128, 64, 128, 256, 512, 256, 128], [64], [128, 256], [512]], #r1 FLOP > 2
        #[[256], [128, 64, 128, 256, 512, 256, 128], [64, 128], [256], [512]], #r1 FLOP > 2
        #[[256], [128, 64, 128, 256, 512, 256, 128, 64], [128], [256], [512]], #r1 FLOP > 2
        [[256, 128], [64], [128], [256, 512, 256, 128, 64, 128, 256], [512]], #r1
        [[256, 128], [64], [128, 256], [512, 256, 128, 64, 128, 256], [512]], #r1
        [[256, 128], [64], [128, 256, 512], [256, 128, 64, 128, 256], [512]], #r1
        [[256, 128], [64], [128, 256, 512, 256], [128, 64, 128, 256], [512]], #r1
        [[256, 128], [64], [128, 256, 512, 256, 128], [64, 128, 256], [512]], #r1
        [[256, 128], [64], [128, 256, 512, 256, 128, 64], [128, 256], [512]], #r1
        [[256, 128], [64], [128, 256, 512, 256, 128, 64, 128], [256], [512]], #r1
        [[256, 128], [64, 128], [256], [512, 256, 128, 64, 128, 256], [512]], #r1
        [[256, 128], [64, 128], [256, 512], [256, 128, 64, 128, 256], [512]], #r1
        [[256, 128], [64, 128], [256, 512, 256], [128, 64, 128, 256], [512]], #r1
        [[256, 128], [64, 128], [256, 512, 256, 128], [64, 128, 256], [512]], #r1
        [[256, 128], [64, 128], [256, 512, 256, 128, 64], [128, 256], [512]], #r1
        [[256, 128], [64, 128], [256, 512, 256, 128, 64, 128], [256], [512]], #r1
        [[256, 128], [64, 128, 256], [512], [256, 128, 64, 128, 256], [512]], #r1
        [[256, 128], [64, 128, 256], [512, 256], [128, 64, 128, 256], [512]], #r1
        [[256, 128], [64, 128, 256], [512, 256, 128], [64, 128, 256], [512]], #r1
        [[256, 128], [64, 128, 256], [512, 256, 128, 64], [128, 256], [512]], #r1
        [[256, 128], [64, 128, 256], [512, 256, 128, 64, 128], [256], [512]], #r1
        #[[256, 128], [64, 128, 256, 512], [256], [128, 64, 128, 256], [512]], #r1 FLOP > 2
        #[[256, 128], [64, 128, 256, 512], [256, 128], [64, 128, 256], [512]], #r1 FLOP > 2
        #[[256, 128], [64, 128, 256, 512], [256, 128, 64], [128, 256], [512]], #r1 FLOP > 2
        #[[256, 128], [64, 128, 256, 512], [256, 128, 64, 128], [256], [512]], #r1 FLOP > 2
        #[[256, 128], [64, 128, 256, 512, 256], [128], [64, 128, 256], [512]], #r1 FLOP > 2
        #[[256, 128], [64, 128, 256, 512, 256], [128, 64], [128, 256], [512]], #r1 FLOP > 2
        #[[256, 128], [64, 128, 256, 512, 256], [128, 64, 128], [256], [512]], #r1 FLOP > 2
        #[[256, 128], [64, 128, 256, 512, 256, 128], [64], [128, 256], [512]], #r1 FLOP > 2
        #[[256, 128], [64, 128, 256, 512, 256, 128], [64, 128], [256], [512]], #r1 FLOP > 2
        #[[256, 128], [64, 128, 256, 512, 256, 128, 64], [128], [256], [512]], #r1 FLOP > 2
    ]
    large_cf = [
        [[256], [128], [64], [128, 256, 512, 256, 128, 64, 128, 256, 512, 256, 128, 64, 128, 256], [512]],# r3
        [[256], [128], [64, 128], [256, 512, 256, 128, 64, 128, 256, 512, 256, 128, 64, 128, 256], [512]],# r3
        [[256], [128, 64], [128], [256, 512, 256, 128, 64, 128, 256, 512, 256, 128, 64, 128, 256], [512]],# r3
        [[256], [128, 64], [128, 256], [512, 256, 128, 64, 128, 256, 512, 256, 128, 64, 128, 256], [512]],# r3
        [[256], [128], [64, 128, 256], [512, 256, 128, 64, 128, 256, 512, 256, 128, 64, 128, 256], [512]],# r3
        [[256], [128], [64, 128, 256, 512], [256, 128, 64, 128, 256, 512, 256, 128, 64, 128, 256], [512]],# r3
        #[[256], [128], [64, 128, 256, 512, 256], [128, 64, 128, 256, 512, 256, 128, 64, 128, 256], [512]],# r2  GMac > 2.5
        #[[256], [128], [64, 128, 256, 512, 256, 128], [64, 128, 256, 512, 256, 128, 64, 128, 256], [512]],# r2  GMac > 2.5
        #[[256], [128], [64, 128, 256, 512, 256, 128, 64], [128, 256, 512, 256, 128, 64, 128, 256], [512]],# r2  GMac > 2.5
        #[[256], [128], [64, 128, 256, 512, 256, 128, 64, 128], [256, 512, 256, 128, 64, 128, 256], [512]],# r2  GMac > 2.5
        #[[256], [128], [64, 128, 256, 512, 256, 128, 64, 128, 256], [512, 256, 128, 64, 128, 256], [512]],# r1 FLOPs > 1.5
        #[[256], [128], [64, 128, 256, 512, 256, 128, 64, 128, 256, 512], [256, 128, 64, 128, 256], [512]],# r1 FLOPs > 1.5
        #[[256], [128], [64, 128, 256, 512, 256, 128, 64, 128, 256, 512, 256], [128, 64, 128, 256], [512]],# r1 FLOPs > 1.5
        #[[256], [128], [64, 128, 256, 512, 256, 128, 64, 128, 256, 512, 256, 128], [64, 128, 256], [512]],# r1 FLOPs > 1.5
        #[[256], [128], [64, 128, 256, 512, 256, 128, 64, 128, 256, 512, 256, 128, 64], [128, 256], [512]],# r1 FLOPs > 1.5
        #[[256], [128], [64, 128, 256, 512, 256, 128, 64, 128, 256, 512, 256, 128, 64, 128], [256], [512]],# r1 FLOPs > 1.5
        [[256], [128, 64], [128, 256, 512], [256, 128, 64, 128, 256, 512, 256, 128, 64, 128, 256], [512]],# r3
        #[[256], [128, 64], [128, 256, 512, 256], [128, 64, 128, 256, 512, 256, 128, 64, 128, 256], [512]],# r2 GMac > 2.5
        #[[256], [128, 64], [128, 256, 512, 256, 128], [64, 128, 256, 512, 256, 128, 64, 128, 256], [512]],# r2 GMac > 2.5
        #[[256], [128, 64], [128, 256, 512, 256, 128, 64], [128, 256, 512, 256, 128, 64, 128, 256], [512]],# r2 GMac > 2.5
        #[[256], [128, 64], [128, 256, 512, 256, 128, 64, 128], [256, 512, 256, 128, 64, 128, 256], [512]],# r2 GMac > 2.5
        #[[256], [128, 64], [128, 256, 512, 256, 128, 64, 128, 256], [512, 256, 128, 64, 128, 256], [512]],# r2 GMac > 2.5
        #[[256], [128, 64], [128, 256, 512, 256, 128, 64, 128, 256, 512], [256, 128, 64, 128, 256], [512]],# r1  FLOPs > 1.5
        #[[256], [128, 64], [128, 256, 512, 256, 128, 64, 128, 256, 512, 256], [128, 64, 128, 256], [512]],# r1 FLOPs > 1.5
        #[[256], [128, 64], [128, 256, 512, 256, 128, 64, 128, 256, 512, 256, 128], [64, 128, 256], [512]],# r1 FLOPs > 1.5
        #[[256], [128, 64], [128, 256, 512, 256, 128, 64, 128, 256, 512, 256, 128, 64], [128, 256], [512]],# r1 FLOPs > 1.5
        #[[256], [128, 64], [128, 256, 512, 256, 128, 64, 128, 256, 512, 256, 128, 64, 128], [256], [512]],# r1 FLOPs > 1.5
    ]
    channel_options = {
        'basic': [[256 ,128], [64], [128], [256], [512]],
        # 'small': [[256], [128, 64, 128], [256, 512, 256 ,128], [64, 128, 256], [512]],
        'small': get_cf(small_cf, cf_index),
        'large': get_cf(large_cf, cf_index),
    }

    channels = channel_options.get(typesize, channel_options['basic'])


    print(f'THE ACTUAL CHANNEL: {typesize}')
    print(f'THE cf_index: {cf_index}')
    print(f'THE config: {channels}')

    if cifar:
        in_size = (32, 32)
        init_conv_stride = 1
        strides = [1, 1, 2, 2, 2]
    else:
        in_size = (224, 224)
        init_conv_stride = 2
        strides = [1, 2, 2, 2, 2]
        # if typesize == 'basic':
        #     strides = [1, 2, 2, 2, 2]
        # elif typesize == 'large':
        #     strides = [1, 2, 2, 2, 2]
        # else:
        #     strides = [2, 1, 2, 2, 2]
    
    return SpatialTickNet(
        num_classes=num_classes,
        init_conv_channels=init_conv_channels,
        init_conv_stride=init_conv_stride,
        channels=channels,
        strides=strides,
        in_size=in_size
    )
