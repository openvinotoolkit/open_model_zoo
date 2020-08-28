# resnet-50-tf

## Use Case and High-Level Description

`resnet-50-tf` is a TensorFlow\* implementation of ResNet-50 - an image classification model
pretrained on the ImageNet dataset. Originally redistributed in Saved model format,
converted to frozen graph using `tf.graph_util` module.
For details see [paper](https://arxiv.org/abs/1512.03385),
[repository](https://github.com/tensorflow/models/tree/v2.2.0/official/r1/resnet).

### Steps to Reproduce Conversion to Frozen Graph

1. Install TensorFlow\*, version 1.14.0.
2. Download [pretrained weights](http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v1_fp32_savedmodel_NHWC_jpg.tar.gz)
3. Run example conversion code, avaliable at [freeze_saved_model.py](./freeze_saved_model.py)
```sh
python3 freeze_saved_model.py --saved_model_dir path/to/downloaded/saved_model --save_file path/to/resulting/frozen_graph.pb
```

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 8.2164        |
| MParams           | 25.53         |
| Source framework  | TensorFlow\*  |

## Accuracy

| Metric | Original model | Converted model |
| ------ | -------------- | --------------- |
| Top 1  | 76.45%          | 76.17%          |
| Top 5  | 93.05%          | 92.98%           |

## Performance

## Input

### Original Model

Image, name: `map/TensorArrayStack/TensorArrayGatherV3`,  shape: `1,224,224,3`, format is `B,H,W,C` where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `RGB`.
Mean values: [123.68,116.78,103.94].

### Converted Model

Image, name: `map/TensorArrayStack/TensorArrayGatherV3`,  shape: `1,224,224,3`, format is `B,H,W,C` where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `BGR`.

## Output

### Original Model

Object classifier according to ImageNet classes, name: `softmax_tensor`,  shape: `1,1001`, output data format is `B,C` where:

- `B` - batch size
- `C` - predicted probabilities for each class in  [0, 1] range

### Converted Model

Object classifier according to ImageNet classes, name: `softmax_tensor`,  shape: `1,1001`, output data format is `B,C` where:

- `B` - batch size
- `C` - predicted probabilities for each class in  [0, 1] range

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/tensorflow/models/master/LICENSE).
A copy of the license is provided in [APACHE-2.0-TensorFlow.txt](../licenses/APACHE-2.0-TensorFlow.txt).
