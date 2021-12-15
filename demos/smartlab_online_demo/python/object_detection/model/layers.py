import torch
from torch import nn
from torch import Tensor
from typing import Callable,Optional

class SiLU(nn.Module):
  # original source: https://github.com/Megvii-BaseDetection/YOLOX
  # Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
  """export-friendly version of nn.SiLU()"""
  @staticmethod
  def forward(x):
    return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
  # original source: https://github.com/Megvii-BaseDetection/YOLOX
  # Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
  if name == "silu":
      module = nn.SiLU(inplace=inplace)
  elif name == "relu":
      module = nn.ReLU(inplace=inplace)
  elif name == "lrelu":
      module = nn.LeakyReLU(0.1, inplace=inplace)
  else:
      raise AttributeError("Unsupported act type: {}".format(name))
  return module

class BaseConv(nn.Module):
  # original source: https://github.com/Megvii-BaseDetection/YOLOX
  # Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
  
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
  # original source: https://github.com/Megvii-BaseDetection/YOLOX
  # Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
  
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class ResLayer(nn.Module):
  # original source: https://github.com/Megvii-BaseDetection/YOLOX
  # Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
  
  "Residual layer with `in_channels` inputs."
  def __init__(self, in_channels: int):
    super().__init__()
    mid_channels = in_channels // 2
    self.layer1 = BaseConv(
      in_channels, mid_channels, ksize=1, stride=1, act="lrelu"
    )
    self.layer2 = BaseConv(
      mid_channels, in_channels, ksize=3, stride=1, act="lrelu"
    )
  
  def forward(self, x):
    out = self.layer2(self.layer1(x))
    return x + out


class Bottleneck(nn.Module):
  # original source: https://github.com/Megvii-BaseDetection/YOLOX
  # Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
  
  # Standard bottleneck
  def __init__(
    self,
    in_channels,
    out_channels,
    shortcut=True,
    expansion=0.5,
    depthwise=False,
    act="silu",
  ):
    super().__init__()
    hidden_channels = int(out_channels * expansion)
    Conv = DWConv if depthwise else BaseConv
    self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
    self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
    self.use_add = shortcut and in_channels == out_channels
  
  def forward(self, x):
    y = self.conv2(self.conv1(x))
    if self.use_add:
      y = y + x
    return y


class SPPBottleneck(nn.Module):
  # original source: https://github.com/Megvii-BaseDetection/YOLOX
  # Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
  """Spatial pyramid pooling layer used in YOLOv3-SPP"""

  def __init__(
    self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
  ):
    super().__init__()
    hidden_channels = in_channels // 2
    self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
    self.m = nn.ModuleList(
      [
        nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
        for ks in kernel_sizes
      ]
    )
    conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
    self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

  def forward(self, x):
    x = self.conv1(x)
    x = torch.cat([x] + [m(x) for m in self.m], dim=1)
    x = self.conv2(x)
    return x


class CSPLayer(nn.Module):
  # original source: https://github.com/Megvii-BaseDetection/YOLOX
  # Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
  """C3 in yolov5, CSP Bottleneck with 3 convolutions"""
  def __init__(
    self,
    in_channels,
    out_channels,
    n=1,
    shortcut=True,
    expansion=0.5,
    depthwise=False,
    act="silu",
  ):
    """
    Args:
      in_channels (int): input channels.
      out_channels (int): output channels.
      n (int): number of Bottlenecks. Default value: 1.
    """
    # ch_in, ch_out, number, shortcut, groups, expansion
    super().__init__()
    hidden_channels = int(out_channels * expansion)  # hidden channels
    self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
    self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
    self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
    module_list = [
      Bottleneck(
        hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
      )
      for _ in range(n)
    ]
    self.m = nn.Sequential(*module_list)

  def forward(self, x):
    x_1 = self.conv1(x)
    x_2 = self.conv2(x)
    x_1 = self.m(x_1)
    x = torch.cat((x_1, x_2), dim=1)
    return self.conv3(x)


class Focus(nn.Module):
  # original source: https://github.com/Megvii-BaseDetection/YOLOX
  # Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
  
  """Focus width and height information into channel space."""
  def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
    super().__init__()
    self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

  def forward(self, x):
    # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
    patch_top_left = x[..., ::2, ::2]
    patch_top_right = x[..., ::2, 1::2]
    patch_bot_left = x[..., 1::2, ::2]
    patch_bot_right = x[..., 1::2, 1::2]
    x = torch.cat(
      (
        patch_top_left,
        patch_bot_left,
        patch_top_right,
        patch_bot_right,
      ),
      dim=1,
    )
    return self.conv(x)


### end

'''
### duplicate nameing  resnet & yolox

class Bottleneck(nn.Module):
  # original source: https://github.com/Megvii-BaseDetection/YOLOX
  # Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
  
  # Standard bottleneck
  def __init__(
    self,
    in_channels,
    out_channels,
    shortcut=True,
    expansion=0.5,
    depthwise=False,
    act="silu",
  ):
    super().__init__()
    hidden_channels = int(out_channels * expansion)
    Conv = DWConv if depthwise else BaseConv
    self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
    self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
    self.use_add = shortcut and in_channels == out_channels
  
  def forward(self, x):
    y = self.conv2(self.conv1(x))
    if self.use_add:
      y = y + x
    return y
'''

class BasicBlock(nn.Module):
  """Basic Block for resnet 18 and resnet 34
  """
  #BasicBlock and BottleNeck block
  #have different output size
  #we use class attribute expansion
  #to distinct
  expansion = 1
  def __init__(self, in_channels, out_channels, stride=1):
    super().__init__()
    #residual function
    self.residual_function = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_channels * BasicBlock.expansion)
    )
    #shortcut
    self.shortcut = nn.Sequential()
    #the shortcut output dimension is not the same with residual function
    #use 1*1 convolution to match the dimension
    if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels * BasicBlock.expansion)
      )

  def forward(self, x):
      return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """
    Residual block for resnet over 50 layers
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
      super().__init__()
      self.residual_function = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels * BottleNeck.expansion),
      )
      self.shortcut = nn.Sequential()
      if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion)
        )

    def forward(self, x):
      return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))