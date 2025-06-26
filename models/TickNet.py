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
    basic_cf = [
        [[256], [128], [64], [128], [256, 512]], # config: 0, GMac = , Flop = , acc:
        [[256], [128], [64], [128, 256], [512]], # config: 1, GMac = , Flop = , acc: 57.81%
        [[256], [128], [64, 128], [256], [512]], # config: 2, GMac = , Flop = , acc:
        [[256], [128, 64], [128], [256], [512]], # config: 3, GMac = , Flop = , acc:
        [[256, 128], [64], [128], [256], [512]], # config: 4, GMac = , Flop = , acc:

        [[256], [128], [64], [128], [256, 512]], # config: 5, GMac = , Flop = , acc:
        [[256], [128], [64], [128, 256], [512]], # config: 6, GMac = , Flop = , acc: ?
        [[256], [128], [64, 128], [256], [512]], # config: 7, GMac = , Flop = , acc:
        [[256], [128, 64], [128], [256], [512]], # config: 8, GMac = , Flop = , acc:
        [[256, 128], [64], [128], [256], [512]], # config: 9, GMac = , Flop = , acc:
        #[[256]<- stride =1, [128]<- stride=2, [64], [128, 256], [512]], # GMac(), Flop (), config 1: 53.99%
    ]
    small_cf = [
        [[256], [128], [64], [128, 256, 512, 256, 128, 64, 128, 256], [512]],#0  ,FLOP: 0.775 G ,#r5: 27.28% -> config 0:
        [[256], [128], [64, 128], [256, 512, 256, 128, 64, 128, 256], [512]],#1  ,FLOP: 0.785 G ,#r5: 31.16% -> config 1:
        [[256], [128], [64, 128, 256], [512, 256, 128, 64, 128, 256], [512]],#2  ,FLOP: 0.825 G ,#r5: 32.58% -> candidate 0: config:2 -> 64.59%
        [[256], [128], [64, 128, 256, 512], [256, 128, 64, 128, 256], [512]],#3  ,FLOP: 0.985 G ,#r5: 31.66% -> candidate 1: config:3 -> 64,95%
        [[256], [128], [64, 128, 256, 512, 256], [128, 64, 128, 256], [512]],#4  ,FLOP: 1.08 G  ,#r4 GMac > 2.1 -> config 2:
        [[256], [128], [64, 128, 256, 512, 256, 128], [64, 128, 256], [512]],#5  ,FLOP: 1.105 G ,#r4 GMac > 2.1 -> config 5:
        [[256], [128], [64, 128, 256, 512, 256, 128, 64], [128, 256], [512]],#6  ,FLOP: 1.115 G ,#r4 GMac > 2.1 -> config 6:
        [[256], [128], [64, 128, 256, 512, 256, 128, 64, 128], [256], [512]],#7  ,FLOP: 1.125 G ,#r4 GMac > 2.1 -> config 7:
        [[256], [128, 64], [128], [256, 512, 256, 128, 64, 128, 256], [512]],#8  ,FLOP: 0.815 G ,#r5: 29.85%    -> config 8:
        [[256], [128, 64], [128, 256], [512, 256, 128, 64, 128, 256], [512]],#9  ,FLOP: 0.855 G ,#r5: 30.64% -> candidate 2: config:9 -> 64.58%
        [[256], [128, 64], [128, 256, 512], [256, 128, 64, 128, 256], [512]],#10 ,FLOP: 1.01 G  ,#r4 GMac > 2.1 -> config 10:
        [[256], [128, 64], [128, 256, 512, 256], [128, 64, 128, 256], [512]],#11 ,FLOP: 1.11 G  ,#r4 GMac > 2.1, FLOP: 1.11G -> config 11: -> 66.34%
        [[256], [128, 64], [128, 256, 512, 256, 128], [64, 128, 256], [512]],#12 ,FLOP: 1.135 G ,#r4 GMac > 2.1 -> config 12:
        [[256], [128, 64], [128, 256, 512, 256, 128, 64], [128, 256], [512]],#13 ,FLOP: 1.14 G  ,#r3 GMac > 2.4 -> config 13:
        [[256], [128, 64], [128, 256, 512, 256, 128, 64, 128], [256], [512]],#14 ,FLOP: 1.15 G  ,#r3 GMac > 2.4 -> config 14:
        [[256], [128, 64, 128], [256], [512, 256, 128, 64, 128, 256], [512]],#15 ,FLOP: 0.895 G ,#r5: 28.28%    -> config 15:
        [[256], [128, 64, 128], [256, 512], [256, 128, 64, 128, 256], [512]],#16 ,FLOP: 1.05 G  ,#r4 GMac > 2.1 -> config 16:
        [[256], [128, 64, 128], [256, 512, 256], [128, 64, 128, 256], [512]],#17 ,FLOP: 1.15 G  ,#r3 GMac > 2.4 -> config 17: Current Best FLOP 1.21 (67.05)
        [[256], [128, 64, 128], [256, 512, 256, 128], [64, 128, 256], [512]],#18 ,FLOP: 1.175 G ,#r3 GMac > 2.4 -> config 18:
        [[256], [128, 64, 128], [256, 512, 256, 128, 64], [128, 256], [512]],#19 ,FLOP: 1.18 G  ,#r3 GMac > 2.4 -> config 19:
        [[256], [128, 64, 128], [256, 512, 256, 128, 64, 128], [256], [512]],#20 ,FLOP: 1.19 G  ,#r3 GMac > 2.4 -> config 20:
        [[256], [128, 64, 128, 256], [512], [256, 128, 64, 128, 256], [512]],#21 ,FLOP: 1.21 G  ,#r3 GMac > 2.4 -> config 21:
        [[256], [128, 64, 128, 256], [512, 256], [128, 64, 128, 256], [512]],#22 ,FLOP: 1.31 G  ,#r3 GMac > 2.4 -> config 22:
        [[256], [128, 64, 128, 256], [512, 256, 128], [64, 128, 256], [512]],#23 ,FLOP: 1.335 G ,#r3 GMac > 2.4 -> config 23:
        [[256], [128, 64, 128, 256], [512, 256, 128, 64], [128, 256], [512]],#24 ,FLOP: 1.34 G  ,#r3 GMac > 2.4 -> config 24:
        [[256], [128, 64, 128, 256], [512, 256, 128, 64, 128], [256], [512]],#25 ,FLOP: 1.35 G  ,#r3 GMac > 2.4 -> config 25:
        [[256], [128, 64, 128, 256, 512], [256], [128, 64, 128, 256], [512]],#26 ,FLOP: 1.935 G ,#r2 GMac > 3 -> config 26:
        [[256], [128, 64, 128, 256, 512], [256, 128], [64, 128, 256], [512]],#27 ,FLOP: 1.96 G  ,#r1 FLOP > 2 -> config 27:
        [[256], [128, 64, 128, 256, 512], [256, 128, 64], [128, 256], [512]], #r1 FLOP > 2 -> config 28:
        [[256], [128, 64, 128, 256, 512], [256, 128, 64, 128], [256], [512]], #r1 FLOP > 2 -> config 29:
        [[256], [128, 64, 128, 256, 512, 256], [128], [64, 128, 256], [512]], #r1 FLOP > 2 -> config 30:
        [[256], [128, 64, 128, 256, 512, 256], [128, 64], [128, 256], [512]], #r1 FLOP > 2 -> config 31:
        [[256], [128, 64, 128, 256, 512, 256], [128, 64, 128], [256], [512]], #r1 FLOP > 2 -> config 32:
        [[256], [128, 64, 128, 256, 512, 256, 128], [64], [128, 256], [512]], #r1 FLOP > 2 -> config 33:
        [[256], [128, 64, 128, 256, 512, 256, 128], [64, 128], [256], [512]], #r1 FLOP > 2 -> config 34:
        [[256], [128, 64, 128, 256, 512, 256, 128, 64], [128], [256], [512]], #r1 FLOP > 2 -> config 35:
        [[256, 128], [64], [128], [256, 512, 256, 128, 64, 128, 256], [512]], #r3 GMac > 2.4 -> config 36:
        [[256, 128], [64], [128, 256], [512, 256, 128, 64, 128, 256], [512]], #r3 GMac > 2.4 -> config 37:
        [[256, 128], [64], [128, 256, 512], [256, 128, 64, 128, 256], [512]], #r3 GMac > 2.4 -> config 38:
        [[256, 128], [64], [128, 256, 512, 256], [128, 64, 128, 256], [512]], #r2 GMac > 3
        [[256, 128], [64], [128, 256, 512, 256, 128], [64, 128, 256], [512]], #r2 GMac > 3
        [[256, 128], [64], [128, 256, 512, 256, 128, 64], [128, 256], [512]], #r2 GMac > 3
        [[256, 128], [64], [128, 256, 512, 256, 128, 64, 128], [256], [512]], #r2 GMac > 3
        [[256, 128], [64, 128], [256], [512, 256, 128, 64, 128, 256], [512]], #r3 GMac > 2.4 -> config 43:
        [[256, 128], [64, 128], [256, 512], [256, 128, 64, 128, 256], [512]], #r2 GMac > 3
        [[256, 128], [64, 128], [256, 512, 256], [128, 64, 128, 256], [512]], #r2 GMac > 3
        [[256, 128], [64, 128], [256, 512, 256, 128], [64, 128, 256], [512]], #r2 GMac > 3
        [[256, 128], [64, 128], [256, 512, 256, 128, 64], [128, 256], [512]], #r2 GMac > 3
        [[256, 128], [64, 128], [256, 512, 256, 128, 64, 128], [256], [512]], #r2 GMac > 3
        [[256, 128], [64, 128, 256], [512], [256, 128, 64, 128, 256], [512]], #r2 GMac > 3
        [[256, 128], [64, 128, 256], [512, 256], [128, 64, 128, 256], [512]], #r2 GMac > 3
        [[256, 128], [64, 128, 256], [512, 256, 128], [64, 128, 256], [512]], #r2 GMac > 3
        [[256, 128], [64, 128, 256], [512, 256, 128, 64], [128, 256], [512]], #r2 GMac > 3
        [[256, 128], [64, 128, 256], [512, 256, 128, 64, 128], [256], [512]], #r2 GMac > 3
        [[256, 128], [64, 128, 256, 512], [256], [128, 64, 128, 256], [512]], #r1 FLOP > 2
        [[256, 128], [64, 128, 256, 512], [256, 128], [64, 128, 256], [512]], #r1 FLOP > 2
        [[256, 128], [64, 128, 256, 512], [256, 128, 64], [128, 256], [512]], #r1 FLOP > 2
        [[256, 128], [64, 128, 256, 512], [256, 128, 64, 128], [256], [512]], #r1 FLOP > 2
        [[256, 128], [64, 128, 256, 512, 256], [128], [64, 128, 256], [512]], #r1 FLOP > 2
        [[256, 128], [64, 128, 256, 512, 256], [128, 64], [128, 256], [512]], #r1 FLOP > 2
        [[256, 128], [64, 128, 256, 512, 256], [128, 64, 128], [256], [512]], #r1 FLOP > 2
        [[256, 128], [64, 128, 256, 512, 256, 128], [64], [128, 256], [512]], #r1 FLOP > 2
        [[256, 128], [64, 128, 256, 512, 256, 128], [64, 128], [256], [512]], #r1 FLOP > 2
        [[256, 128], [64, 128, 256, 512, 256, 128, 64], [128], [256], [512]], #r1 FLOP > 2
    ]
    large_cf = [
        [[256], [128], [64], [128, 256, 512, 256, 128, 64, 128, 256, 512, 256, 128, 64, 128, 256], [512]],#0  ,FLOP: 0.89 G  ,r3 ,19.05% (20e), config: 0 -> 61,06%
        [[256], [128], [64, 128], [256, 512, 256, 128, 64, 128, 256, 512, 256, 128, 64, 128, 256], [512]],#1  ,FLOP: 0.9 G   ,r3 ,20.45% (20e), config: 1 -> 62,5%
        [[256], [128, 64], [128], [256, 512, 256, 128, 64, 128, 256, 512, 256, 128, 64, 128, 256], [512]],#2  ,FLOP: 0.925 G ,r3 ,18.56%      , config: 1 -> 62,5%
        [[256], [128, 64], [128, 256], [512, 256, 128, 64, 128, 256, 512, 256, 128, 64, 128, 256], [512]],#3  ,FLOP: 0.965 G ,r3 ,26.13% (20e), config: 3 -> 64,34%/64.06%/64,38%
        [[256], [128], [64, 128, 256], [512, 256, 128, 64, 128, 256, 512, 256, 128, 64, 128, 256], [512]],#4  ,FLOP: 0.94 G  ,r3 ,19.58%      , config: 1 -> 62,5%
        [[256], [128], [64, 128, 256, 512], [256, 128, 64, 128, 256, 512, 256, 128, 64, 128, 256], [512]],#5  ,FLOP: 1.095 G ,r3 ,22.97% (20e), config: 5 -> 64,17%
        [[256], [128], [64, 128, 256, 512, 256], [128, 64, 128, 256, 512, 256, 128, 64, 128, 256], [512]],#6  ,FLOP: 1.195 G ,r2 ,GMac > 2.5
        [[256], [128], [64, 128, 256, 512, 256, 128], [64, 128, 256, 512, 256, 128, 64, 128, 256], [512]],#7  ,FLOP: 1.22 G  ,r2 ,GMac > 2.5
        [[256], [128], [64, 128, 256, 512, 256, 128, 64], [128, 256, 512, 256, 128, 64, 128, 256], [512]],#8  ,FLOP: 1.225 G ,r2 ,GMac > 2.5
        [[256], [128], [64, 128, 256, 512, 256, 128, 64, 128], [256, 512, 256, 128, 64, 128, 256], [512]],#9  ,FLOP: 1.235 G ,r2 ,GMac > 2.5
        [[256], [128], [64, 128, 256, 512, 256, 128, 64, 128, 256], [512, 256, 128, 64, 128, 256], [512]],#10 ,FLOP: 1.275 G ,r1 ,FLOPs > 1.5
        [[256], [128], [64, 128, 256, 512, 256, 128, 64, 128, 256, 512], [256, 128, 64, 128, 256], [512]],#11 ,FLOP: 1.435 G ,r1 ,FLOPs > 1.5
        [[256], [128], [64, 128, 256, 512, 256, 128, 64, 128, 256, 512, 256], [128, 64, 128, 256], [512]],#12 ,FLOP: 1.53 G  ,r1 ,FLOPs > 1.5
        [[256], [128], [64, 128, 256, 512, 256, 128, 64, 128, 256, 512, 256, 128], [64, 128, 256], [512]],#13 ,FLOP: 1.555 G ,r1 ,FLOPs > 1.5
        [[256], [128], [64, 128, 256, 512, 256, 128, 64, 128, 256, 512, 256, 128, 64], [128, 256], [512]],#14 ,FLOP: 1.565 G ,r1 ,FLOPs > 1.5
        [[256], [128], [64, 128, 256, 512, 256, 128, 64, 128, 256, 512, 256, 128, 64, 128], [256], [512]],#15 ,FLOP: 1.575 G ,r1 ,FLOPs > 1.5
        [[256], [128, 64], [128, 256, 512], [256, 128, 64, 128, 256, 512, 256, 128, 64, 128, 256], [512]],#16 ,FLOP: 1.12 G  ,r3 ,21.49%
        [[256], [128, 64], [128, 256, 512, 256], [128, 64, 128, 256, 512, 256, 128, 64, 128, 256], [512]],#17 ,FLOP: 1.22 G  ,r2 ,GMac > 2.5
        [[256], [128, 64], [128, 256, 512, 256, 128], [64, 128, 256, 512, 256, 128, 64, 128, 256], [512]],#18 ,FLOP: 1.245 G ,r2 ,GMac > 2.5
        [[256], [128, 64], [128, 256, 512, 256, 128, 64], [128, 256, 512, 256, 128, 64, 128, 256], [512]],#19 ,FLOP: 1.255 G ,r2 ,GMac > 2.5
        [[256], [128, 64], [128, 256, 512, 256, 128, 64, 128], [256, 512, 256, 128, 64, 128, 256], [512]],#20 ,FLOP: 1.265 G ,r2 ,GMac > 2.5
        [[256], [128, 64], [128, 256, 512, 256, 128, 64, 128, 256], [512, 256, 128, 64, 128, 256], [512]],#21 ,FLOP: 1.305 G ,r2 ,GMac > 2.5
        [[256], [128, 64], [128, 256, 512, 256, 128, 64, 128, 256, 512], [256, 128, 64, 128, 256], [512]],#22 ,FLOP: 1.46 G  ,r1 ,FLOPs > 1.5
        [[256], [128, 64], [128, 256, 512, 256, 128, 64, 128, 256, 512, 256], [128, 64, 128, 256], [512]],#23 ,FLOP: 1.56 G  ,r1 ,FLOPs > 1.5
        [[256], [128, 64], [128, 256, 512, 256, 128, 64, 128, 256, 512, 256, 128], [64, 128, 256], [512]],#24 ,FLOP: 1.585 G ,r1 ,FLOPs > 1.5
        [[256], [128, 64], [128, 256, 512, 256, 128, 64, 128, 256, 512, 256, 128, 64], [128, 256], [512]],#25 ,FLOP: 1.59 G  ,r1 ,FLOPs > 1.5
        [[256], [128, 64], [128, 256, 512, 256, 128, 64, 128, 256, 512, 256, 128, 64, 128], [256], [512]],#26 ,FLOP: 1.6 G   ,r1 ,FLOPs > 1.5
        [[256, 128], [64, 128, 256, 512], [256, 128, 64, 128, 256, 512], [256, 128, 64, 128, 256], [512]] #27 ,FLOP: 2.69 G  ,r0 ,config: 27 -> OOM (16GB)% (200e) MACs (5.5), FLOPs(2.75GB)
    ]
    channel_options = {
        'basic': get_cf(basic_cf, cf_index),
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
        if typesize == 'basic' and cf_index < 5:
            strides = [2, 1, 2, 2, 2]
        else:
            strides = [1, 2, 2, 2, 2]
    
    return SpatialTickNet(
        num_classes=num_classes,
        init_conv_channels=init_conv_channels,
        init_conv_stride=init_conv_stride,
        channels=channels,
        strides=strides,
        in_size=in_size
    )
