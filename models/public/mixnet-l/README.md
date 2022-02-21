# mixnet-l

## Use Case and High-Level Description
MixNets are a family of mobile-sizes image classification models equipped with MixConv,
a new type of mixed depthwise convolutions. There are three MixNet architectures -
`MixNet-S` (Small), `MixNet-M` (Middle), `MixNet-L` (Large). The main differences are using
MixConv with different kernel sizes and number of layers. Using `MixNet-L` allows to achieve greater accuracy.
All the MixNet models have been pretrained on the ImageNet dataset.
For details about this family of models, check out the [TensorFlow Cloud TPU repository](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet) and [paper](https://arxiv.org/abs/1907.09595).

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 0.565         |
| MParams           | 7.300         |
| Source framework  | TensorFlow\*  |

## Accuracy

| Metric | Original model | Converted model |
| ------ | -------------- | --------------- |
| Top 1  | 78.30%         | 78.30%          |
| Top 5  | 93.91%         | 93.91%          |

## Input

### Original Model

Image, name - `image`,  shape - `1, 224, 224, 3`, format is `B, H, W, C`, where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `RGB`.

### Converted Model

Image, name - `IteratorGetNext/placeholder_out_port_0`,  shape - `1, 224, 224, 3`, format is `B, H, W, C`, where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `BGR`.

## Output

### Original Model

Object classifier according to ImageNet classes, name - `logits`,  shape - `1,1000`, output data format is `B,C` where:

- `B` - batch size
- `C` - predicted logits for each class

### Converted Model

Object classifier according to ImageNet classes, name - `logits`,  shape - `1,1000`, output data format is `B,C` where:

- `B` - batch size
- `C` - predicted logits for each class

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
[Apache License, Version 2.0](https://raw.githubusercontent.com/tensorflow/tpu/master/LICENSE).
A copy of the license is provided in `<omz_dir>/models/public/licenses/APACHE-2.0-TF-TPU.txt`.
