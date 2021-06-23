# yolof

## Use Case and High-Level Description

YOLOF is a simple, fast, and efficient object detector without FPN. Model based on ["You Only Look One-level Feature"](https://arxiv.org/abs/2103.09460) paper. It was implemented in PyTorch\* framework. For details see [repository](https://github.com/megvii-model/YOLOF). This model was pre-trained on [Common Objects in Context (COCO)](https://cocodataset.org/#home) dataset with 80 classes.

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Detection     |
| GFLOPs            | ?             |
| MParams           | ?             |
| Source framework  | PyTorch\*     |

## Accuracy

Accuracy metrics obtained on [Common Objects in Context (COCO)](https://cocodataset.org/#home) validation dataset for converted model.

| Metric                                                                | Value  |
| --------------------------------------------------------------------- | -------|
| mAP                                                                   | 59.33% |
| [COCO mAP (0.5)](https://cocodataset.org/#detection-eval)             | 66.12% |
| [COCO mAP (0.5:0.05:0.95)](https://cocodataset.org/#detection-eval)   | 42.99% |

## Input

### Original model

Image, name - `image_input`, shape - `1, 3, 608, 608`, format is `B, H, W, C`, where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `BGR`.
Mean values - [103.53, 116.28, 123.675].

### Converted model

Image, name - `image_input`, shape - `1, 3, 608, 608`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original model



### Converted model

1. The array of detection summary info, name - `boxes`, shape - `1, 510, 38, 38`. The anchor values are `16,16,  32,32,  64,64,  128,128,  256,256,  512,512`.

For each case format is `B,N*85,Cx,Cy`, where
- `B` - batch size
- `N` - number of detection boxes for cell
- `Cx`, `Cy` - cell index

Detection box has format [`x`,`y`,`h`,`w`,`box_score`,`class_no_1`, ..., `class_no_80`], where:
- (`x`,`y`) - coordinates of box center relative to the cell
- `h`,`w` - raw height and width of box, apply [exponential function](https://en.wikipedia.org/wiki/Exponential_function) and multiply by corresponding anchors to get absolute height and width values
- `box_score` - confidence of detection box in [0,1] range
- `class_no_1`,...,`class_no_80` - probability distribution over the classes in the [0,1] range, multiply by confidence value to get confidence of each class


The model was trained on Microsoft\* COCO dataset version with 80 categories of object. Mapping to class names provided in `<omz_dir>/data/dataset_classes/coco_80cl.txt` file.

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
[license](https://raw.githubusercontent.com/megvii-model/YOLOF/main/LICENSE):

```
MIT License

Copyright (c) 2021 megvii-model

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
