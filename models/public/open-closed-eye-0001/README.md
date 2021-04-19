# open-closed-eye-0001

## Use Case and High-Level Description

Fully convolutional network for recognition of eye state ('open', 'closed').

## Example

![](./assets/1.png)
![](./assets/2.png)
![](./assets/3.png)

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Source framework                | PyTorch\*                                 |
| GFlops                          | 0.0014                                    |
| MParams                         | 0.0113                                    |
| Accuracy                        | 95.84%                                    |

## Dataset

[MRL Eye Dataset](http://mrl.cs.vsb.cz/eyedataset) is used for training and validation (each 10th image is used for test).

## Inputs

Image, name: `input.1`, shape: `1, 3, 32, 32` in `1, C, H, W` format, where:

- `C` - channel
- `H` - height
- `W` - width

Expected color order is `BGR`.

## Outputs

Name: `19`, shape: `1, 2, 1, 1` - Softmax output across 2 type classes: [open, closed]


## Download a Model and Convert it into Inference Engine Format

You can download models and if necessary convert them into Inference Engine format using the [Model Downloader and other automation tools](../../../tools/downloader/README.md) as shown in the examples below.

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
