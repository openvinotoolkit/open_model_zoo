# alexnet

## Use Case and High-Level Description

The `alexnet` model is designed to perform image classification. Just like other common classification models, the `alexnet` model has been pretrained on the ImageNet image database. For details about this model, check out the [paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).

The model input is a blob that consists of a single image of 1x3x227x227 in BGR order. The BGR mean values need to be subtracted as follows: [104, 117, 123] before passing the image blob into the network.

The model output for `alexnet` is the usual object classifier output for the 1000 different classifications matching those in the ImageNet database.

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 1.5           |
| MParams           | 60.965        |
| Source framework  | Caffe\*         |

## Accuracy

See [the original model's documentation](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet).

## Performance

## Input

### Original model

Image, name - `data`, shape - `1,3,227,227`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.
Mean values - [104, 117, 123]

### Converted model

### Original model

Image, name - `data`, shape - `1,3,227,227`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original model

Object classifier according to ImageNet classes, name - `prob`, shape - `1,1000`, output data format is `B,C` where:

- `B` - batch size
- `C` - Predicted probabilities for each class in  [0, 1] range

### Converted model

Object classifier according to ImageNet classes, name - `prob`, shape - `1,1000`, output data format is `B,C` where:

- `B` - batch size
- `C` - Predicted probabilities for each class in  [0, 1] range


## Legal Information

The original model is distributed under the following
[license](https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/readme.md):

```
This model is released for unrestricted use.
```
