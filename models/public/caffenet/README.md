# caffenet

## Use Case and High-Level Description

CaffeNet\* model is used for classification. For details see [paper](https://arxiv.org/abs/1408.5093).

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Classification                            |
| GFlops                          | 1.463                                     |
| MParams                         | 60.965                                    |
| Source framework                | Caffe\*                                   |

## Accuracy

| Metric | Value  |
| ------ | ------ |
| Top 1  | 56.714%|
| Top 5  | 79.916%|

## Input

### Original Model

Image, name: `data`, shape: `1, 3, 227, 227`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: `BGR`.
Mean values: [104.0, 117.0, 123.0].

### Converted Model

Image, name: `data`, shape: `1, 3, 227, 227`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: `BGR`.

## Output

### Original Model

Object classifier according to ImageNet classes, name: `prob`,  shape: `1, 1000`. Contains predicted
probability for each class.

### Converted model

Object classifier according to ImageNet classes, name: `prob`,  shape: `1, 1000`. Contains predicted
probability for each class.

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

The original model is distributed under the following
[license](https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_reference_caffenet/readme.md):

```
This model is released for unrestricted use.
```
