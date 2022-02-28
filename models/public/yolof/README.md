# yolof

## Use Case and High-Level Description

YOLOF is a simple, fast, and efficient object detector without FPN. Model based on ["You Only Look One-level Feature"](https://arxiv.org/abs/2103.09460) paper. It was implemented in PyTorch\* framework. Model used `DarkNet-53` with Cross Stage Partial blocks as backbone. For details see [repository](https://github.com/megvii-model/YOLOF). This model was pre-trained on [Common Objects in Context (COCO)](https://cocodataset.org/#home) dataset with 80 classes. Mapping of class IDs to label names provided in `<omz_dir>/data/dataset_classes/coco_80cl.txt` file.

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Detection     |
| GFLOPs            | 175.37942     |
| MParams           | 48.228        |
| Source framework  | PyTorch\*     |

## Accuracy

Accuracy metrics obtained on [Common Objects in Context (COCO)](https://cocodataset.org/#home) validation dataset for converted model.

| Metric                                                                | Value  |
| --------------------------------------------------------------------- | -------|
| mAP                                                                   | 60.69% |
| [COCO mAP (0.5)](https://cocodataset.org/#detection-eval)             | 66.23% |
| [COCO mAP (0.5:0.05:0.95)](https://cocodataset.org/#detection-eval)   | 43.63% |

## Input

### Original model

Image, name - `image_input`, shape - `1, 3, 608, 608`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

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

The list of instances. The postprocessing is implemented inside the model and is performed while inference of model. So each instance it is a object with next fields:
- `detection box`
- `label` - predicted class ID
- `score` - onfidence for the predicted class

Detection box has format [`x_min`, `y_min`, `x_max`, `y_max`], where:

- (`x_min`, `y_min`) - coordinates of the top left bounding box corner
- (`x_max`, `y_max`) - coordinates of the bottom right bounding box corner

### Converted model

The array of detection summary info, name - `boxes`, shape - `1, 504, 38, 38`. The anchor values are `16,16,  32,32,  64,64,  128,128,  256,256,  512,512`.

For each case format is `B, N*84, Cx, Cy`, where

- `B` - batch size
- `Cx`, `Cy` - cell index
- `N` - number of detection boxes for cell

Detection box has format [`x`, `y`, `h`, `w`, `class_id_1`, ..., `class_id_80`], where:

- (`x`, `y`) - raw coordinates of box center, multiply by corresponding anchors to get relative to the cell coordinates
- `h`, `w` - raw height and width of box, apply [exponential function](https://en.wikipedia.org/wiki/Exponential_function) and multiply by corresponding anchors to get absolute height and width values
- `class_id_1`,...,`class_id_80` - probability distribution over the classes in logits format, apply [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) to get confidence of each class


## Download a Model and Convert it into OpenVINO™ IR Format

You can download models and if necessary convert them into OpenVINO™ IR format using the [Model Downloader and other automation tools](../../../tools/model_tools/README.md) as shown in the examples below.

An example of using the Model Downloader:
```
omz_downloader --name <model_name>
```

An example of using the Model Converter:
```
omz_converter --name <model_name>
```

## Demo usage

The model can be used in the following demos provided by the Open Model Zoo to show its capabilities:

* [Object Detection C++ Demo](../../../demos/object_detection_demo/cpp/README.md)
* [Object Detection Python\* Demo](../../../demos/object_detection_demo/python/README.md)

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
