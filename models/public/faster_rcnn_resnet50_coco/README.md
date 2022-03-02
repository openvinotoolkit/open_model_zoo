# faster_rcnn_resnet50_coco

## Use Case and High-Level Description

Faster R-CNN ResNet-50 model. Used for object detection. For details, see the [paper](https://arxiv.org/abs/1506.01497).

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Object detection                          |
| GFlops                          | 57.203                                    |
| MParams                         | 29.162                                    |
| Source framework                | TensorFlow\*                              |

## Accuracy

| Metric         | Value |
| -------------- | ----- |
| coco_precision | 31.09%|

## Input

### Original Model

Image, name: `image_tensor`, shape: `1, 600, 1024, 3`, format: `B, H, W, C`, where:

- `B` - batch size
- `H` - image height
- `W` - image width
- `C` - number of channels

Expected color order: `RGB`.

### Converted Model

1. Image, name: `image_tensor`, shape: `1, 600, 1024, 3`, format: `B, H, W, C`, where:

- `B` - batch size
- `H` - image height
- `W` - image width
- `C` - number of channels

    Expected color order: `BGR`.

2. Information of input image size, name: `image_info`, shape: `1, 3`, format: `B, C`, where:

    - `B` - batch size
    - `C` - vector of 3 values in format `H, W, S`, where `H` is an image height, `W` is an image width, `S` is an image scale factor (usually 1).

## Output

### Original Model

1. Classifier, name: `detection_classes`. Contains predicted bounding boxes classes in a range [1, 91]. The model was trained on [Common Objects in Context (COCO)](https://cocodataset.org/#home) dataset version with 91 categories of objects, 0 class is for background. Mapping to class names provided in `<omz_dir>/data/dataset_classes/coco_91cl_bkgr.txt` file
2. Probability, name: `detection_scores`. Contains probability of detected bounding boxes.
3. Detection box, name: `detection_boxes`. Contains detection boxes coordinates in format `[y_min, x_min, y_max, x_max]`, where (`x_min`, `y_min`)  are coordinates top left corner, (`x_max`, `y_max`) are coordinates right bottom corner. Coordinates are rescaled to input image size.
4. Detections number, name: `num_detections`. Contains the number of predicted detection boxes.

### Converted Model

The array of summary detection information, name: `reshape_do_2d`, shape: `1, 1, 100, 7` in the format `1, 1, N, 7`, where `N` is the number of detected
bounding boxes. For each detection, the description has the format:
[`image_id`, `label`, `conf`, `x_min`, `y_min`, `x_max`, `y_max`], where:

- `image_id` - ID of the image in the batch
- `label` - predicted class ID in range [1, 91], mapping to class names provided in `<omz_dir>/data/dataset_classes/coco_91cl_bkgr.txt` file
- `conf` - confidence for the predicted class
- (`x_min`, `y_min`) - coordinates of the top left bounding box corner (coordinates stored in normalized format, in range [0, 1])
- (`x_max`, `y_max`) - coordinates of the bottom right bounding box corner  (coordinates stored in normalized format, in range [0, 1])

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

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/tensorflow/models/master/LICENSE).
A copy of the license is provided in `<omz_dir>/models/public/licenses/APACHE-2.0-TF-Models.txt`.
