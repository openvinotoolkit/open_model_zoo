# googlenet-v4

## Use Case and High-Level Description

The `googlenet-v4` model is the most recent of the [Inception](https://arxiv.org/pdf/1602.07261.pdf) family of models designed to perform image classification. Like the other Inception models, the `googlenet-v4` model has been pretrained on the ImageNet image database. For details about this family of models, check out the paper.

The model input is a blob that consists of a single image of 1x3x299x299 in BGR order. The BGR mean values need to be subtracted as follows: [128.0,128.0,128.0] before passing the image blob into the network. In addition, values must be divided by 0.0078125.

The model output for `googlenet-v4` is the typical object classifier output for the 1000 different classifications matching those in the ImageNet database.

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 24.584        |
| MParams           | 42.648        |
| Source framework  | Caffe\*         |

## Accuracy

See [https://github.com/soeaver/caffe-model](https://github.com/soeaver/caffe-model).

## Performance

## Input

### Original model

Image, name - `data`, shape - `1,3,299,299`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.
Mean values - [128.0,128.0,128.0], scale value - 128.0

### Converted model

Image,  name - `data`, shape - `1,3,299,299`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`

## Output

### Original model

Object classifier according to ImageNet classes, name - `prob`,  shape - `1,1000`, output data format is `B,C` where:

- `B` - batch size
- `C` - Predicted probabilities for each class in  [0, 1] range

### Converted model

Object classifier according to ImageNet classes, name - `prob`,  shape - `1,1000`, output data format is `B,C` where:

- `B` - batch size
- `C` - Predicted probabilities for each class in  [0, 1] range

## Legal Information

[https://raw.githubusercontent.com/soeaver/caffe-model/master/LICENSE]()
