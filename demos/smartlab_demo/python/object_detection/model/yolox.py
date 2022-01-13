import torch.nn as nn
from model.yolox_layers import YOLOXHead
from model.necks import YOLOPAFPN


class YOLOX(nn.Module):
  # original source: https://github.com/Megvii-BaseDetection/YOLOX
  # Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
      super().__init__()
      self.backbone = YOLOPAFPN() if backbone is None else backbone
      self.head = YOLOXHead(80) if head is None else head

    def forward(self, x, targets=None):
      # fpn output content features of [dark3, dark4, dark5]
      fpn_outs = self.backbone(x)
      if self.training:
        assert targets is not None
        loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
          fpn_outs, targets, x
        )
        outputs = {
          "total_loss": loss,
          "iou_loss": iou_loss,
          "l1_loss": l1_loss,
          "conf_loss": conf_loss,
          "cls_loss": cls_loss,
          "num_fg": num_fg,
        }
      else:
          outputs = self.head(fpn_outs)
      return outputs
