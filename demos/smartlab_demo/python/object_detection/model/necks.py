import torch
import torch.nn as nn
from model.backbones import Darknet, CSPDarknet
from model.layers import BaseConv, CSPLayer, DWConv


class YOLOFPN(nn.Module):
  # original source: https://github.com/Megvii-BaseDetection/YOLOX
  # Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
  """
  YOLOFPN module. Darknet 53 is the default backbone of this model.
  """
  def __init__(
    self,
    depth=53,
    in_features=["dark3", "dark4", "dark5"],
  ):
    super().__init__()
    self.backbone = Darknet(depth)
    self.in_features = in_features
    # out 1
    self.out1_cbl = self._make_cbl(512, 256, 1)
    self.out1 = self._make_embedding([256, 512], 512 + 256)
    # out 2
    self.out2_cbl = self._make_cbl(256, 128, 1)
    self.out2 = self._make_embedding([128, 256], 256 + 128)
    # upsample
    self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

  def _make_cbl(self, _in, _out, ks):
    return BaseConv(_in, _out, ks, stride=1, act="lrelu")

  def _make_embedding(self, filters_list, in_filters):
    m = nn.Sequential(
      *[
        self._make_cbl(in_filters, filters_list[0], 1),
        self._make_cbl(filters_list[0], filters_list[1], 3),
        self._make_cbl(filters_list[1], filters_list[0], 1),
        self._make_cbl(filters_list[0], filters_list[1], 3),
        self._make_cbl(filters_list[1], filters_list[0], 1),
      ]
    )
    return m

  def load_pretrained_model(self, filename="./weights/darknet53.mix.pth"):
    with open(filename, "rb") as f:
      state_dict = torch.load(f, map_location="cpu")
    print("loading pretrained weights...")
    self.backbone.load_state_dict(state_dict)

  def forward(self, inputs):
    """
    Args:
      inputs (Tensor): input image.
    Returns:
      Tuple[Tensor]: FPN output features..
    """
    #  backbone
    out_features = self.backbone(inputs)
    x2, x1, x0 = [out_features[f] for f in self.in_features]
    #  yolo branch 1
    x1_in = self.out1_cbl(x0)
    x1_in = self.upsample(x1_in)
    x1_in = torch.cat([x1_in, x1], 1)
    out_dark4 = self.out1(x1_in)
    #  yolo branch 2
    x2_in = self.out2_cbl(out_dark4)
    x2_in = self.upsample(x2_in)
    x2_in = torch.cat([x2_in, x2], 1)
    out_dark3 = self.out2(x2_in)
    outputs = (out_dark3, out_dark4, x0)
    return outputs


class YOLOPAFPN(nn.Module):
  # original source: https://github.com/Megvii-BaseDetection/YOLOX
  # Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
  """
  YOLOv3 neck subnet. Darknet 53 is the default backbone of this model.
  """
  def __init__(
    self,
    depth=1.0,
    width=1.0,
    in_features=("dark3", "dark4", "dark5"),
    in_channels=[256, 512, 1024],
    depthwise=False,
    act="silu",
  ):
    super().__init__()
    self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
    self.in_features = in_features
    self.in_channels = in_channels
    Conv = DWConv if depthwise else BaseConv
    self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
    self.lateral_conv0 = BaseConv(
      int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
    )
    self.C3_p4 = CSPLayer(
      int(2 * in_channels[1] * width),
      int(in_channels[1] * width),
      round(3 * depth),
      False,
      depthwise=depthwise,
      act=act,
    )  # cat
    self.reduce_conv1 = BaseConv(
      int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
    )
    self.C3_p3 = CSPLayer(
      int(2 * in_channels[0] * width),
      int(in_channels[0] * width),
      round(3 * depth),
      False,
      depthwise=depthwise,
      act=act,
    )
    # bottom-up conv
    self.bu_conv2 = Conv(
      int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
    )
    self.C3_n3 = CSPLayer(
      int(2 * in_channels[0] * width),
      int(in_channels[1] * width),
      round(3 * depth),
      False,
      depthwise=depthwise,
      act=act,
    )
    # bottom-up conv
    self.bu_conv1 = Conv(
      int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
    )
    self.C3_n4 = CSPLayer(
      int(2 * in_channels[1] * width),
      int(in_channels[2] * width),
      round(3 * depth),
      False,
      depthwise=depthwise,
      act=act,
    )

  def forward(self, input):
    """
    Args:
      inputs: input images.
    Returns:
      Tuple[Tensor]: FPN feature.
    """
    #  backbone
    out_features = self.backbone(input)
    features = [out_features[f] for f in self.in_features]
    [x2, x1, x0] = features
    fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
    f_out0 = self.upsample(fpn_out0)  # 512/16
    f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
    f_out0 = self.C3_p4(f_out0)  # 1024->512/16
    fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
    f_out1 = self.upsample(fpn_out1)  # 256/8
    f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
    pan_out2 = self.C3_p3(f_out1)  # 512->256/8
    p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
    p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
    pan_out1 = self.C3_n3(p_out1)  # 512->512/16
    p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
    p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
    pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32
    outputs = (pan_out2, pan_out1, pan_out0)
    return outputs

### end