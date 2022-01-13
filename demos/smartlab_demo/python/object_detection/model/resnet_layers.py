import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import get_activation


class resnet_headerlayers(nn.Module):
  """
  Classification Heads
  """
  def __init__(
      self,
      num_classes: int,
      blockexpansion: int = 4,
      act : str = 'relu'
      ):
    super().__init__()
    self.fc1 = nn.Linear(512 * blockexpansion,1024)
    self.fc2 = nn.Linear(1024,num_classes)
    self.act1 = get_activation(act,inplace = True)
    self.softmax = nn.Softmax()

  def forward(self,x):
    x = torch.flatten(x, 1)
    # x = F.relu(self.fc1(x)) #changes
    x = self.fc1(x)
    x = self.act1(x)
    x = self.fc2(x)
    x = self.softmax(x)
    return x 