import numpy as np

class Calculate_Unique(object):
  def __init__(self, char_input):
    self._shape = np.array([-1])
    self._char_input = char_input.reshape(self._shape)

  def _Unique_(self):
    x = self._char_input
    y = []
    idx = []
    for i in range(len(x)):
      if x[i] not in y:
        y.append(x[i])
    for i in range(len(x)):
      for index, result in enumerate(x[i] == y):
        if result == True:
          idx.append(index)
          break
    return y, idx

  def _Unique(self):
    y, idx = self._Unique_()
    shape = len(self._char_input)
    if len(y) >= shape:
      return y, idx
    _y = []
    _y.extend(y)
    for j in range(shape-len(y)):
      _y.append(y[len(y)-1])
    return _y, idx
