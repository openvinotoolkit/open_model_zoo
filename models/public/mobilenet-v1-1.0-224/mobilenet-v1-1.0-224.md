# mobilenet-v1-1.0-224

## Use Case and High-Level Description

`mobilenet-v1-1.0-224` is one of [MobileNet V1 architecture](https://arxiv.org/abs/1704.04861) with the width multiplier 1.0 and resolution 224. It is small, low-latency, low-power models parameterized to meet the resource constraints of a variety of use cases. They can be built upon for classification, detection, embeddings and segmentation similar to how other popular large scale models are used.

## Example

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Classification                            |
| GFlops                          | 1.148                                     |
| MParams                         | 4.221                                     |
| Source framework                | Caffe\*                              |

## Performance

## Input

### Original model

Image, name - `input` , shape - `1,3,224,224`, format `B,C,H,W`, where:

    - B - batch size
    - C - number of channels
    - H - image height
    - W - image width

   Expected color order - BGR.
   Mean values - [103.94,116.78,123.68], scale factor for each channel - 58.8235294117647

### Converted model

Image, name - `input` , shape - `1,3,224,224`, format `B,C,H,W`, where:

    - B - batch size
    - C - number of channels
    - H - image height
    - W - image width

   Expected color order - BGR.

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

[LICENSE](https://raw.githubusercontent.com/shicai/MobileNet-Caffe/26a8b8c0afb6114a07c1c9e4f550e4e0dd8cced1/LICENSE)
