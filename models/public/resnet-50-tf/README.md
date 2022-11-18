# resnet-50-tf

## Use Case and High-Level Description

`resnet-50-tf` is a TensorFlow\* implementation of ResNet-50 - an image classification model
pre-trained on the ImageNet dataset. Originally redistributed in Saved model format,
converted to frozen graph using `tf.graph_util` module.
For details see [paper](https://arxiv.org/abs/1512.03385),
[repository](https://github.com/tensorflow/models/tree/v2.2.0/official/r1/resnet).

### Steps to Reproduce Conversion to Frozen Graph

1. Install TensorFlow\*, version 1.14.0.
2. Download [pre-trained weights](http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v1_fp32_savedmodel_NHWC_jpg.tar.gz)
3. Run example conversion code, available at `<omz_dir>/models/public/resnet-50-tf/freeze_saved_model.py`
```sh
python3 freeze_saved_model.py --saved_model_dir path/to/downloaded/saved_model --save_file path/to/resulting/frozen_graph.pb
```

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
| Top 1  | 76.45%          | 76.17%         |
| Top 5  | 93.05%          | 92.98%         |

## Input

### Original Model

Image, name: `map/TensorArrayStack/TensorArrayGatherV3`,  shape: `1, 224, 224, 3`, format is `B, H, W, C`, where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `RGB`.
Mean values: [123.68, 116.78, 103.94].

### Converted Model

Image, name: `map/TensorArrayStack/TensorArrayGatherV3`,  shape: `1, 224, 224, 3`, format is `B, H, W, C`, where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `BGR`.

## Output

### Original Model

Object classifier according to ImageNet classes, name: `softmax_tensor`,  shape: `1, 1001`, output data format is `B, C`, where:

- `B` - batch size
- `C` - predicted probabilities for each class in  [0, 1] range

### Converted Model

Object classifier according to ImageNet classes, name: `softmax_tensor`,  shape: `1, 1001`, output data format is `B, C`, where:

- `B` - batch size
- `C` - predicted probabilities for each class in  [0, 1] range

## Download a Model and Convert it into OpenVINO™ IR Format

You can download models and if necessary convert them into OpenVINO™ IR format using the [Model Downloader and other automation tools](../../../tools/model_tools/README.md) as shown in the examples below.

An example of using the Model Downloader:
```
omz_downloader --name <model_name>
```

An example of using the Model Converter:
```
omz_converter --name <model_name>
```

## Demo usage

The model can be used in the following demos provided by the Open Model Zoo to show its capabilities:

* [Classification Benchmark C++ Demo](../../../demos/classification_benchmark_demo/cpp/README.md)
* [Classification Python\* Demo](../../../demos/classification_demo/python/README.md)

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/tensorflow/models/master/LICENSE).
A copy of the license is provided in `<omz_dir>/models/public/licenses/APACHE-2.0-TF-Models.txt`.
