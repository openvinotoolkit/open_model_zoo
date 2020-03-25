# googlenet-v1-tf

## Use Case and High-Level Description

The `googlenet-v1-tf` model is one of the Inception family, designed to perform image classification.
Like the other Inception models, the `googlenet-v1-tf` model has been pretrained on the ImageNet image database.
Originally redistributed as a checkpoint file, was converted to frozen graph.
For details about this family of models, check out the [paper](https://arxiv.org/abs/1602.07261), [repository](https://github.com/tensorflow/models/tree/master/research/slim).

### Steps to Reproduce Conversion to Frozen Graph

1. Clone the original repository
```sh
git clone https://github.com/tensorflow/models.git
cd models/research/slim
```
2. Checkout the commit that the conversion was tested on:
```sh
git checkout 5d36f19
```
3. Apply `freeze.py.patch` patch
```sh
git apply path/to/freeze.py.patch
```
4. Download the [pretrained weights](http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz)
5. Install the dependencies:
```sh
pip install tensorflow==1.14.0
```
6. Run
```sh
python3 freeze.py --ckpt path/to/inception_v1.ckpt --name inception_v1 --num_classes 1001 --output InceptionV1/Logits/Predictions/Softmax
```

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 3.016         |
| MParams           | 6.619         |
| Source framework  | TensorFlow\*  |

## Accuracy

| Metric | Original model | Converted model |
| ------ | -------------- | --------------- |
| Top 1  | 69.81          | 69.81           |
| Top 5  | 89.61          | 89.61           |

## Performance

## Input

### Original model

Image, name - `input`, shape - `1,224,224,3`, format is `B,H,W,C` where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `RGB`.
Mean values - [127.5, 127.5, 127.5], scale value - 127.5

### Converted model

Image,  name - `data`, shape - `1,3,224,224`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`

## Output

### Original model

Object classifier according to ImageNet classes, name - `InceptionV1/Logits/Predictions/Softmax`,  shape - `1,1001`, output data format is `B,C` where:

- `B` - batch size
- `C` - Predicted probabilities for each class in  [0, 1] range

### Converted model

Object classifier according to ImageNet classes, name - `InceptionV1/Logits/Predictions/Softmax`,  shape - `1,1001`, output data format is `B,C` where:

- `B` - batch size
- `C` - Predicted probabilities for each class in  [0, 1] range

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://github.com/tensorflow/models/blob/master/LICENSE).
A copy of the license is provided in [APACHE-2.0-TensorFlow.txt](../licenses/APACHE-2.0-TensorFlow.txt).
