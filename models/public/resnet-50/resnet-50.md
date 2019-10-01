# resnet-50

## Use Case and High-Level Description

[ResNet-50](https://arxiv.org/pdf/1512.03385.pdf)

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 6.996         |
| MParams           | 25.53         |
| Source framework  | Caffe\*       |

## Accuracy

## Performance

## Input

### Original Model

Image, name: `data`,  shape: `1,3,224,224`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`. 
Mean values: [104, 117, 123].

### Converted Model

Image, name: `data`,  shape: `1,3,224,224`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original Model

Object classifier according to ImageNet classes, name: `prob`,  shape: `1,1000`, output data format is `B,C` where:

- `B` - batch size
- `C` - predicted probabilities for each class in  [0, 1] range

### Converted Model

Object classifier according to ImageNet classes, name: `prob`,  shape: `1,1000`, output data format is `B,C` where:

- `B` - batch size
- `C` - predicted probabilities for each class in  [0, 1] range

## Legal Information

[LICENSE](https://raw.githubusercontent.com/KaimingHe/deep-residual-networks/master/LICENSE)