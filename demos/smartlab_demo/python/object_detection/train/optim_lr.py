import math
from functools import partial


class LRScheduler:
  # modified version of: https://github.com/Megvii-BaseDetection/YOLOX
  # Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
  def __init__(self, name, lr, iters_per_epoch, total_epochs, **kwargs):
    """
    Supported lr schedulers: [cos, warmcos, multistep]
    Args:
      lr (float): learning rate.
      iters_per_peoch (int): number of iterations in one epoch.
      total_epochs (int): number of epochs in training.
      kwargs (dict):
        - cos: None
        - warmcos: [warmup_epochs, warmup_lr_start (default 1e-6)]
        - multistep: [milestones (epochs), gamma (default 0.1)]
    """
    self.lr = lr
    self.iters_per_epoch = iters_per_epoch
    self.total_epochs = total_epochs
    self.total_iters = iters_per_epoch * total_epochs ### 总循环步数是总epoch乘以每个epoch的循环步数
    self.__dict__.update(kwargs)
    self.lr_func = self._get_lr_func(name)

  def update_lr(self, iters): ### 根据 循环步idx（不是epoch） 给出 lr 值
    return self.lr_func(iters)

  def _get_lr_func(self, name):
    ### 利用 getattr + default 可以优先采用设定的参数值
    ### 若未定义，也可以给出一个默认参数
    if name == "cos":  ### 用partial预先填好参数，只暴露最后一个参数iters
      lr_func = partial(cos_lr, self.lr, self.total_iters)
    elif name == "warmcos":
      warmup_total_iters = self.iters_per_epoch * self.warmup_epochs
      warmup_lr_start = getattr(self, "warmup_lr_start", 1e-6)
      lr_func = partial(
        warm_cos_lr,
        self.lr,
        self.total_iters,
        warmup_total_iters,
        warmup_lr_start,
      )
    elif name == "yoloxwarmcos":
      warmup_total_iters = self.iters_per_epoch * self.warmup_epochs
      no_aug_iters = self.iters_per_epoch * self.no_aug_epochs
      warmup_lr_start = getattr(self, "warmup_lr_start", 0)
      min_lr_ratio = getattr(self, "min_lr_ratio", 0.2)
      lr_func = partial(
        yolox_warm_cos_lr,
        self.lr,
        min_lr_ratio,
        self.total_iters,
        warmup_total_iters,
        warmup_lr_start,
        no_aug_iters,
      )
    elif name == "yoloxsemiwarmcos":
      warmup_lr_start = getattr(self, "warmup_lr_start", 0)
      min_lr_ratio = getattr(self, "min_lr_ratio", 0.2)
      warmup_total_iters = self.iters_per_epoch * self.warmup_epochs
      no_aug_iters = self.iters_per_epoch * self.no_aug_epochs
      normal_iters = self.iters_per_epoch * self.semi_epoch
      semi_iters = self.iters_per_epoch_semi * (
        self.total_epochs - self.semi_epoch - self.no_aug_epochs
      )
      lr_func = partial(
        yolox_semi_warm_cos_lr,
        self.lr,
        min_lr_ratio,
        warmup_lr_start,
        self.total_iters,
        normal_iters,
        no_aug_iters,
        warmup_total_iters,
        semi_iters,
        self.iters_per_epoch,
        self.iters_per_epoch_semi,
      )
    elif name == "multistep":  # stepwise lr schedule
      milestones = [
        int(self.total_iters * milestone / self.total_epochs)
        for milestone in self.milestones
      ]
      gamma = getattr(self, "gamma", 0.1)
      lr_func = partial(multistep_lr, self.lr, milestones, gamma)
    else:
      raise ValueError("Scheduler version {} not supported.".format(name))
    return lr_func


### 按照单位Aplitude的cosine波形改变lr
def cos_lr(lr, total_iters, iters):
  # modified version of: https://github.com/Megvii-BaseDetection/YOLOX
  # Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
  """Cosine learning rate"""
  lr *= 0.5 * (1.0 + math.cos(math.pi * iters / total_iters))
  return lr


### 初期短时快速下降，然后再进入cosine波形
def warm_cos_lr(lr, total_iters, warmup_total_iters, warmup_lr_start, iters):
  # modified version of: https://github.com/Megvii-BaseDetection/YOLOX
  # Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
  """Cosine learning rate with warm up."""
  if iters <= warmup_total_iters:
    lr = (lr - warmup_lr_start) * iters / float(
      warmup_total_iters
    ) + warmup_lr_start
  else:
    lr *= 0.5 * (
      1.0
      + math.cos(
        math.pi
        * (iters - warmup_total_iters)
        / (total_iters - warmup_total_iters)
      )
    )
  return lr


### 类似warm_cos_lr，
### 在初期速降和cosine阶段之间，有一段维持期，保持lr在min_lr不变
def yolox_warm_cos_lr(
  lr,
  min_lr_ratio,
  total_iters,
  warmup_total_iters,
  warmup_lr_start,
  no_aug_iter,
  iters,
):
  # modified version of: https://github.com/Megvii-BaseDetection/YOLOX
  # Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
  """Cosine learning rate with warm up."""
  min_lr = lr * min_lr_ratio
  if iters <= warmup_total_iters:
    # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
    lr = (lr - warmup_lr_start) * pow(
      iters / float(warmup_total_iters), 2
    ) + warmup_lr_start
  elif iters >= total_iters - no_aug_iter:
    lr = min_lr
  else:
    lr = min_lr + 0.5 * (lr - min_lr) * (
      1.0
      + math.cos(
        math.pi
        * (iters - warmup_total_iters)
        / (total_iters - warmup_total_iters - no_aug_iter)
      )
    )
  return lr


### 类似yolox_warm_cos_lr，
### 但是有两个不同的cosine阶段
def yolox_semi_warm_cos_lr(
  lr,
  min_lr_ratio,
  warmup_lr_start,
  total_iters,
  normal_iters,
  no_aug_iters,
  warmup_total_iters,
  semi_iters,
  iters_per_epoch,
  iters_per_epoch_semi,
  iters,
):
  # modified version of: https://github.com/Megvii-BaseDetection/YOLOX
  # Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
  """Cosine learning rate with warm up."""
  min_lr = lr * min_lr_ratio
  if iters <= warmup_total_iters:
    # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
    lr = (lr - warmup_lr_start) * pow(
      iters / float(warmup_total_iters), 2
    ) + warmup_lr_start
  elif iters >= normal_iters + semi_iters:
    lr = min_lr
  elif iters <= normal_iters:
    lr = min_lr + 0.5 * (lr - min_lr) * (
      1.0
      + math.cos(
        math.pi
        * (iters - warmup_total_iters)
        / (total_iters - warmup_total_iters - no_aug_iters)
      )
    )
  else:
    lr = min_lr + 0.5 * (lr - min_lr) * (
      1.0
      + math.cos(
        math.pi
        * (
          normal_iters
          - warmup_total_iters
          + (iters - normal_iters)
          * iters_per_epoch
          * 1.0
          / iters_per_epoch_semi
        )
        / (total_iters - warmup_total_iters - no_aug_iters)
      )
    )
  return lr


### 多阶段lr
def multistep_lr(lr, milestones, gamma, iters):
  # modified version of: https://github.com/Megvii-BaseDetection/YOLOX
  # Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
  """MultiStep learning rate"""
  for milestone in milestones:
    lr *= gamma if iters >= milestone else 1.0
  return lr
