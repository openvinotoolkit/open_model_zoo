# resnet50-binary-0001

## Use Case and High-Level Description

This is a classical classification network for 1000 classes trained on ImageNet.
The difference is that most convolutional layers were replaced by binary once that can be implemented as XNOR+POPCOUN operations.
Only input, final and shortcut layers were kept as FP32, all the rest convolutional layers are replaced by BinaryConvolution layers.


## Specification

| Metric          | Value    |
|-----------------|----------|
| Image size      | 224x224  |
| fp32 conv MFlops| 960      |
| bin conv  MI1ops| 7218     |
| Source framework| PyTorch\*|

## Accuracy

The quality metrics calculated on ImageNet validation dataset is 70.69% accuracy

| Metric                    | Value         |
|---------------------------|---------------|
| Accuracy top-1 (ImageNet) |        70.69% |

## Inputs

A blob with a `BGR` image and the shape `1, 3, 224, 224` in the format `B, C, H, W`, where:

- `B` – batch size
- `C` – number of channels
- `H` – image height
- `W` – image width

It is supposed that input is `BGR` in 0..255 range

## Outputs

The output is a blob with the shape `1, 1000` in the format `B, C`, where:

- `B` - batch size
- `C` - predicted probabilities for each class in [0, 1] range

## Legal Information
[*] Other names and brands may be claimed as the property of others.
