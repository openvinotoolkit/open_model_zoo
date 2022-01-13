import torch
import torch.nn as nn
from model.backbones import ResNet
from model.resnet_layers import resnet_headerlayers
from model.layers import BasicBlock,BottleNeck
class resnet_detectors(nn.Module):
  # modify from: https://github.com/weiaicunzai/pytorch-cifar100
  # Copyright (c) weiaicunzai. All rights reserved.
  def __init__(self, block, num_block, num_classes=100):
    super().__init__()
    self.backbone = ResNet(block, num_block)
    self.head = resnet_headerlayers(num_classes,block.expansion)

  def forward(self,x):
    x = self.backbone(x)
    x = self.head(x)
    return x
  

def resnet18(num_classes):
  """ return a ResNet 18 object
  """
  return resnet_detectors(BasicBlock, [2, 2, 2, 2],num_classes)

def resnet34(num_classes):
  """ return a ResNet 34 object
  """
  return resnet_detectors(BasicBlock, [3, 4, 6, 3],num_classes)

def resnet50(num_classes):
  """ return a ResNet 50 object
  """
  return resnet_detectors(BottleNeck, [3, 4, 6, 3],num_classes)

def resnet101(num_classes):
  """ return a ResNet 101 object
  """
  return resnet_detectors(BottleNeck, [3, 4, 23, 3],num_classes)

def resnet152(num_classes):
  """ return a ResNet 152 object
  """
  return resnet_detectors(BottleNeck, [3, 8, 36, 3],num_classes)
