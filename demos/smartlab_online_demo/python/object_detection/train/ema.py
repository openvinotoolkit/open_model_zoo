"""
 Copyright (C) 2021-2022 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import math
from copy import deepcopy
import torch
import torch.nn as nn

__all__ = ["ModelEMA", "is_parallel"]


def is_parallel(model):
  """check if model is in parallel mode."""
  parallel_type = (
    nn.parallel.DataParallel,
    nn.parallel.DistributedDataParallel,
  )
  return isinstance(model, parallel_type)


class ModelEMA:
  """
  Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
  Keep a moving average of everything in the model state_dict (parameters and buffers).
  This is intended to allow functionality like
  https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
  A smoothed version of the weights is necessary for some training schemes to perform well.
  This class is sensitive where it is initialized in the sequence of model init,
  GPU assignment and distributed training wrappers.
  """
  def __init__(self, model, decay=0.9999, updates=0):
    """
    Args:
      model (nn.Module): model to apply EMA.
      decay (float): ema decay reate.
      updates (int): counter of EMA updates.
    """
    # Create EMA(FP32)
    self.ema = deepcopy(model.module if is_parallel(model) else model).eval()
    self.updates = updates
    # decay exponential ramp (to help early epochs)
    self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
    for p in self.ema.parameters():
      p.requires_grad_(False)

  def update(self, model):
    # Update EMA parameters
    with torch.no_grad():
      self.updates += 1
      d = self.decay(self.updates)
      msd = (
        model.module.state_dict() if is_parallel(model) else model.state_dict()
      )  # model state_dict
      for k, v in self.ema.state_dict().items():
        if v.dtype.is_floating_point:
          v *= d
          v += (1.0 - d) * msd[k].detach()
