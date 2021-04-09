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

## Download a Model and Convert it into Inference Engine Format

You can download models and if necessary convert them into Inference Engine format using the [Model Downloader and other automation tools](../../../tools/downloader/README.md) as shown in the examples below.

An example of using the Model Downloader:
```
python3 <omz_dir>/tools/downloader/downloader.py --name <model_name>
```

An example of using the Model Converter:
```
python3 <omz_dir>/tools/downloader/converter.py --name <model_name>
```

## Legal Information

The original model is distributed under the following
[license](https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_reference_caffenet/readme.md):

```
This model is released for unrestricted use.
```
