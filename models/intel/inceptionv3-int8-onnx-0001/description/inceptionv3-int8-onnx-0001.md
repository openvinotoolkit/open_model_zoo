# inceptionv3-int8-onnx-0001

## Use Case and High-Level Description

This is the Inception v3 model that is designed to perform image classification. The model has been pretrained on the 
ImageNet image database and then symmetrically quantized to INT8 fixed-point precision using Neural Network Compression 
Framework (NNCF).  

The model input is a blob that consists of a single image of "1x3x299x299" in BGR order.

The model output for `inceptionv3-int8-onnx-0001` is the usual object classifier output for the 1000 different 
classifications matching those in the ImageNet database.

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 11.469 |
| MParams           | 23.817 |
| Source framework  | PyTorch    |

## Accuracy

The quality metrics calculated on ImageNet validation dataset is 78.36% accuracy top-1.

| Metric                    | Value         |
|---------------------------|---------------|
| Accuracy top-1 (ImageNet) |         78.36% |

## Performance

## Input

Image, shape - `1,3,299,299`, format is `B,C,H,W` where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `BGR`

## Output

Object classifier according to ImageNet classes, shape -`1,1000`, output data format is `B,C` where:

- `B` - batch size
- `C` - predicted probabilities for each class in  [0, 1] range

