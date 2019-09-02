# squeezenetv1.1-int8-onnx-0001

## Use Case and High-Level Description

This is the SqueezeNet v1.1 model that is designed to perform image classification. 
The model has been pretrained on the ImageNet image database and then symmetrically quantized 
to INT8 fixed-point precision using Neural Network Compression Framework (NNCF).  

The model input is a blob that consists of a single image of "1x3x224x224" in BGR order.

The model output for `squeezenetv1.1-int8-onnx-0001` is the usual object classifier output for the 1000 different classifications matching those in the ImageNet database.

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 0.707 |
| MParams           | 1.236 |
| Source framework  | PyTorch\*    |

## Accuracy

The quality metrics calculated on ImageNet validation dataset is 57.9% accuracy top-1.

| Metric                    | Value         |
|---------------------------|---------------|
| Accuracy top-1 (ImageNet) |         57.9% |

## Performance

## Input

Image, shape - `1,3,224,224`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`

## Output

Object classifier according to ImageNet classes, shape -`1,1000`, output data format is `B,C` where:

- `B` - batch size
- `C` - predicted probabilities for each class in  [0, 1] range

## Legal Information
[*] Other names and brands may be claimed as the property of others.
