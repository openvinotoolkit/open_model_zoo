# open-closed-eye-0001

## Use Case and High-Level Description

Fully convolutional network for recognition of eye state ('open', 'closed').

## Example

![](./1.png)
![](./2.png)
![](./3.png)

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Source framework                | PyTorch*                                  |
| GFlops                          | 0.0014                                    |
| MParams                         | 0.0113                                    |
| Accuracy                        | 95.84%                                     |

## Dataset

[MRL Eye Dataset](http://mrl.cs.vsb.cz/eyedataset) is used for training and validation (each 10th image is used for test).

## Inputs

1. name: "input.1" , shape: [1x3x32x32] - An input image in [1xCxHxW] format. Expected color order is BGR.

## Outputs

1. name: "19", shape: [1, 2, 1, 1] - Softmax output across 2 type classes [open, closed]


## Download a Model and Convert it into Inference Engine Format

You can download models and if necessary convert them into Inference Engine format using the [Model Downloader and other automation tools](../../../../tools/downloader/README.md) as shown in the examples below.

An example of using the Model Downloader:
```
python3 <omz_dir>/tools/downloader/downloader.py --name <model_name>
```

An example of using the Model Converter:
```
python3 <omz_dir>/tools/downloader/converter.py --name <model_name>
```

## Legal Information
[*] Other names and brands may be claimed as the property of others.
