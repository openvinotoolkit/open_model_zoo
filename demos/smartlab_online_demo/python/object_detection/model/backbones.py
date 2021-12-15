#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import torch
import torch.nn as nn
from torch import Tensor
from torch.hub import load_state_dict_from_url
# from .._internally_replaced_utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
from model.layers import BaseConv, CSPLayer, DWConv, Focus, ResLayer, SPPBottleneck

class Darknet(nn.Module):
  # original source: https://github.com/Megvii-BaseDetection/YOLOX
  # Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
  # number of blocks from dark2 to dark5.
  depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}
  def __init__(
    self,
    depth,
    in_channels=3,
    stem_out_channels=32,
    out_features=("dark3", "dark4", "dark5"),
  ):
    """
    Args:
      depth (int): depth of darknet used in model, usually use [21, 53] for this param.
      in_channels (int): number of input channels, for example, use 3 for RGB image.
      stem_out_channels (int): number of output chanels of darknet stem.
        It decides channels of darknet layer2 to layer5.
      out_features (Tuple[str]): desired output layer name.
    """
    super().__init__()
    assert out_features, "please provide output features of Darknet"
    self.out_features = out_features
    self.stem = nn.Sequential(
      BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu"),
      *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2),
    )
    in_channels = stem_out_channels * 2  # 64
    num_blocks = Darknet.depth2blocks[depth]
    # create darknet with `stem_out_channels` and `num_blocks` layers.
    # to make model structure more clear, we don't use `for` statement in python.
    self.dark2 = nn.Sequential(
      *self.make_group_layer(in_channels, num_blocks[0], stride=2)
    )
    in_channels *= 2  # 128
    self.dark3 = nn.Sequential(
      *self.make_group_layer(in_channels, num_blocks[1], stride=2)
    )
    in_channels *= 2  # 256
    self.dark4 = nn.Sequential(
      *self.make_group_layer(in_channels, num_blocks[2], stride=2)
    )
    in_channels *= 2  # 512
    self.dark5 = nn.Sequential(
      *self.make_group_layer(in_channels, num_blocks[3], stride=2),
      *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
    )

  def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1):
    "starts with conv layer then has `num_blocks` `ResLayer`"
    return [
      BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu"),
      *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)],
    ]

  def make_spp_block(self, filters_list, in_filters):
    m = nn.Sequential(
      *[
        BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
        BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
        SPPBottleneck(
          in_channels=filters_list[1],
          out_channels=filters_list[0],
          activation="lrelu",
        ),
        BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
        BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
      ]
    )
    return m

  def forward(self, x):
    outputs = {}
    x = self.stem(x)
    outputs["stem"] = x
    x = self.dark2(x)
    outputs["dark2"] = x
    x = self.dark3(x)
    outputs["dark3"] = x
    x = self.dark4(x)
    outputs["dark4"] = x
    x = self.dark5(x)
    outputs["dark5"] = x
    return {k: v for k, v in outputs.items() if k in self.out_features}


class CSPDarknet(nn.Module):
  # original source: https://github.com/Megvii-BaseDetection/YOLOX
  # Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
  def __init__(
    self,
    dep_mul,
    wid_mul,
    out_features=("dark3", "dark4", "dark5"),
    depthwise=False,
    act="silu",
  ):
    super().__init__()
    assert out_features, "please provide output features of Darknet"
    self.out_features = out_features
    Conv = DWConv if depthwise else BaseConv
    base_channels = int(wid_mul * 64)  # 64
    base_depth = max(round(dep_mul * 3), 1)  # 3
    # stem
    self.stem = Focus(3, base_channels, ksize=3, act=act)
    # dark2
    self.dark2 = nn.Sequential(
      Conv(base_channels, base_channels * 2, 3, 2, act=act),
      CSPLayer(
        base_channels * 2,
        base_channels * 2,
        n=base_depth,
        depthwise=depthwise,
        act=act,
      ),
    )
    # dark3
    self.dark3 = nn.Sequential(
      Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
      CSPLayer(
        base_channels * 4,
        base_channels * 4,
        n=base_depth * 3,
        depthwise=depthwise,
        act=act,
      ),
    )
    # dark4
    self.dark4 = nn.Sequential(
      Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
      CSPLayer(
        base_channels * 8,
        base_channels * 8,
        n=base_depth * 3,
        depthwise=depthwise,
        act=act,
      ),
    )
    # dark5
    self.dark5 = nn.Sequential(
      Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
      SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
      CSPLayer(
        base_channels * 16,
        base_channels * 16,
        n=base_depth,
        shortcut=False,
        depthwise=depthwise,
        act=act,
      ),
    )

  def forward(self, x):
    outputs = {}
    x = self.stem(x)
    outputs["stem"] = x
    x = self.dark2(x)
    outputs["dark2"] = x
    x = self.dark3(x)
    outputs["dark3"] = x
    x = self.dark4(x)
    outputs["dark4"] = x
    x = self.dark5(x)
    outputs["dark5"] = x
    return {k: v for k, v in outputs.items() if k in self.out_features}


class ResNet(nn.Module):
  # original source: https://github.com/weiaicunzai/pytorch-cifar100
  # Copyright (c) weiaicunzai. All rights reserved.

  def __init__(self, block, num_block, num_classes=100):
    super().__init__()

    self.in_channels = 64

    self.conv1 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True))
    #we use a different inputsize than the original paper
    #so conv2_x's stride is 1
    self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
    self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
    self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
    self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
    self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512 * block.expansion, num_classes)

  def _make_layer(self, block, out_channels, num_blocks, stride):
    """make resnet layers(by layer i didnt mean this 'layer' was the
    same as a neuron netowork layer, ex. conv layer), one layer may
    contain more than one residual block
    Args:
        block: block type, basic block or bottle neck block
        out_channels: output depth channel number of this layer
        num_blocks: how many blocks per layer
        stride: the stride of the first block of this layer
    Return:
        return a resnet layer
    """
    # we have num_block blocks per layer, the first block
    # could be 1 or 2, other blocks would always be 1
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_channels, out_channels, stride))
      self.in_channels = out_channels * block.expansion
    return nn.Sequential(*layers)

  def forward(self, x):
    output = self.conv1(x)
    output = self.conv2_x(output)
    output = self.conv3_x(output)
    output = self.conv4_x(output)
    output = self.conv5_x(output)
    output = self.avg_pool(output)
    # output = output.view(output.size(0), -1)
    # output = self.fc(output)
    
    return output

