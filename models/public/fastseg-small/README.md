# fastseg-small

## Use Case and High-Level Description

fastseg-small is an accurate real-time semantic segmentation model, pre-trained on [Cityscapes](https://www.cityscapes-dataset.com) dataset for 19 object classes, listed in `<omz_dir>/data/dataset_classes/cityscapes_19cl.txt` file. See Cityscapes classes [definition](https://www.cityscapes-dataset.com/dataset-overview) for more details. The model was built on MobileNetV3 small backbone and modified segmentation head based on LR-ASPP. This model can be used for efficient segmentation on a variety of real-world street images. For model implementation details see original [repository](https://github.com/ekzhang/fastseg).

## Specification

| Metric            | Value                |
|-------------------|----------------------|
| Type              | Semantic segmentation|
| GOps              | 69.2204              |
| MParams           | 1.1                  |
| Source framework  | PyTorch\*            |

## Accuracy

| Metric    | Value |
| --------- | ----- |
| mean_iou  | 67.15%|

## Input

### Original model

Image, name: `input0`, shape: `1, 3, 1024, 2048`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: `RGB`.
Mean values: [123.675, 116.28, 103.53], scale values: [58.395, 57.12, 57.375]

### Converted Model

Image, name: `input0`, shape: `1, 3, 1024, 2048`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: `BGR`.

## Output

### Original Model

Float values, which represent scores of a predicted class for each image pixel. The model was trained on Cityscapes dataset with 19 categories of objects. Name: `output0`, shape: `1, 19, 1024, 2048` in `B, N, H, W` format, where:

- `B` - batch size
- `N` - number of classes
- `H` - image height
- `W` - image width

### Converted Model

Float values, which represent scores of a predicted class for each image pixel. The model was trained on Cityscapes dataset with 19 categories of objects. Name: `output0`, shape: `1, 19, 1024, 2048` in `B, N, H, W` format, where:

- `B` - batch size
- `N` - number of classes
- `H` - image height
- `W` - image width

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

The original model is distributed under the following
[license](https://raw.githubusercontent.com/ekzhang/fastseg/master/LICENSE.txt):

```
MIT License

Copyright (c) 2020 Eric Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
