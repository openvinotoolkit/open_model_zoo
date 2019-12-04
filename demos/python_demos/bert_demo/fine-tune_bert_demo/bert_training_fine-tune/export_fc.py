from tensorflow.python import pywrap_tensorflow
import os
import numpy as np


checkpoint_path=os.path.join('result2/model.ckpt-468')
reader=pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map=reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    if key == 'output_weights':
      print('tensor_name: ',key)
      weight = reader.get_tensor(key)
    if key == 'output_bias':
      print('tensor_name: ',key)
      bias = reader.get_tensor(key)

np.save('weight.npy',weight)
np.save('bias.npy',bias)
