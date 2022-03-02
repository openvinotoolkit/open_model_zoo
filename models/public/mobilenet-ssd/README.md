# mobilenet-ssd

## Use Case and High-Level Description

The `mobilenet-ssd` model is a Single-Shot multibox Detection (SSD) network intended to perform object detection. This model is implemented using the Caffe\* framework. For details about this model, check out the [repository](https://github.com/chuanqi305/MobileNet-SSD).

The model input is a blob that consists of a single image of `1, 3, 300, 300` in `BGR` order, also like the `densenet-121` model. The BGR mean values need to be subtracted as follows: [127.5, 127.5, 127.5] before passing the image blob into the network. In addition, values must be divided by 0.007843.

The model output is a typical vector containing the tracked object data, as previously described.

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Detection     |
| GFLOPs            | 2.316         |
| MParams           | 5.783         |
| Source framework  | Caffe\*       |

## Accuracy

The accuracy results were obtained on test data from VOC2007 dataset.

| Metric | Value  |
| ------ | ------ |
|  mAP   | 67.00% |

See [the original repository](https://github.com/chuanqi305/MobileNet-SSD).

## Input

### Original model

Image, name - `prob`,  shape - `1, 3, 300, 300`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.
Mean values - [127.5, 127.5, 127.5], scale value - 127.5.

### Converted model

Image, name - `prob`,  shape - `1, 3, 300, 300`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`

## Output

### Original model

The array of detection summary info, name - `detection_out`, shape - `1, 1, 100, 7` in the format `1, 1, N, 7`, where `N` is the number of detected bounding boxes. For each detection, the description has the format:
[`image_id`, `label`, `conf`, `x_min`, `y_min`, `x_max`, `y_max`], where:

- `image_id` - ID of the image in the batch
- `label` - predicted class ID (1..20 - PASCAL VOC defined class ids). Mapping to class names provided by `<omz_dir>/data/dataset_classes/voc_20cl_bkgr.txt` file.
- `conf` - confidence for the predicted class
- (`x_min`, `y_min`) - coordinates of the top left bounding box corner (coordinates are in normalized format, in range [0, 1])
- (`x_max`, `y_max`) - coordinates of the bottom right bounding box corner  (coordinates are in normalized format, in range [0, 1])

### Converted model

The array of detection summary info, name - `detection_out`,  shape - `1, 1, 100, 7` in the format `1, 1, N, 7`, where `N` is the number of detected bounding boxes. For each detection, the description has the format:
[`image_id`, `label`, `conf`, `x_min`, `y_min`, `x_max`, `y_max`], where:

- `image_id` - ID of the image in the batch
- `label` - predicted class ID (1..20 - PASCAL VOC defined class ids). Mapping to class names provided by `<omz_dir>/data/dataset_classes/voc_20cl_bkgr.txt` file.
- `conf` - confidence for the predicted class
- (`x_min`, `y_min`) - coordinates of the top left bounding box corner (coordinates are in normalized format, in range [0, 1])
- (`x_max`, `y_max`) - coordinates of the bottom right bounding box corner  (coordinates are in normalized format, in range [0, 1])

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

* [Single Human Pose Estimation Demo](../../../demos/single_human_pose_estimation_demo/python/README.md)

## Legal Information

The original model is distributed under the following
[license](https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/LICENSE):

```
MIT License

Copyright (c) 2018 chuanqi305

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
