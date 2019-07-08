# squeezenet1.1

## Use Case and High-Level Description

The `squeezenet1.1` updated version of the [SqueezeNet](https://arxiv.org/pdf/1602.07360) topology. It is designed to perform image classification.  It requires 2.4x less computation than [SqueezeNet v1.0](../squeezenet1.0/squeezenet1.0.md) without diminishing accuracy. The SqueezeNet models have been pre-trained on the ImageNet image database. For details about this family of models, check out the [repository](https://github.com/DeepScale/SqueezeNet).

The model input is a blob that consists of a single image of 1x3x227x227 in BGR order. The BGR mean values need to be subtracted as follows: [104, 117, 123] before passing the image blob into the network.

The model output for `squeezenet1.1` is the typical object classifier output for the 1000 different classifications matching those in the ImageNet database.

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 0.785         |
| MParams           | 1.236         |
| Source framework  | Caffe\*         |

## Accuracy

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

[https://raw.githubusercontent.com/DeepScale/SqueezeNet/master/LICENSE]()
