# mobilenet-v2

## Use Case and High-Level Description

[MobileNet V2](https://arxiv.org/pdf/1801.04381.pdf)

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 0.876         |
| MParams           | 3.489         |
| Source framework  | Caffe\*       |

## Accuracy

## Performance

## Input

### Original Model

Image, name: `data`,  shape: `1,3,224,224`, format: `B,C,H,W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.
Mean values: [103.94,116.78,123.68], scale value: 58.8235294117647.

### Converted Model

Image, name: `data`,  shape: `1,3,224,224`, format: `B,C,H,W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original Model

Object classifier according to ImageNet classes, name: `prob`,  shape: `1,1000`, output data format is `B,C`, where:

- `B` - batch size
- `C` - predicted probabilities for each class in a range [0, 1] 

### Converted Model

Object classifier according to ImageNet classes, name: `prob`,  shape: `1,1000`, output data format is `B,C`, where:

- `B` - batch size
- `C` - predicted probabilities for each class in a range [0, 1] 

## Legal Information

[LICENSE](https://raw.githubusercontent.com/shicai/MobileNet-Caffe/26a8b8c0afb6114a07c1c9e4f550e4e0dd8cced1/LICENSE)
