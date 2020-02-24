# resnet18-xnor-binary-onnx-0001

## Use Case and High-Level Description

This is a classical classification network for 1000 classes trained on ImageNet.
The difference is that most convolutional layers were replaced by binary ones that can be implemented as XNOR+POPCOUNT operations.
Only input, final and shortcut layers were kept as FP32, all the rest convolutional layers are replaced by binary convolution layers.


## Specification
| Metric          | Value    |
|-----------------|----------|
| Image size      | 224x224  |
| Source framework  | PyTorch\*             |

## Accuracy

The quality metrics calculated on ImageNet validation dataset is 61.71% accuracy

| Metric                    | Value         |
|---------------------------|---------------|
| Accuracy top-1 (ImageNet) |        61.71% |

## Performance

## Inputs

A blob with a BGR image in the format: [B, C=3, H=224, W=224], where:

- B – batch size
- C – number of channels
- H – image height
- W – image width

It is supposed that input is BGR in 0..255 range

## Outputs

The output is a blob with the shape [B, C=1000].

## Legal Information
[*] Other names and brands may be claimed as the property of others.
