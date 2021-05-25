# ssd_mobilenet_v1_coco

## Use Case and High-Level Description

The `ssd_mobilenet_v1_coco` model is a [Single-Shot multibox Detection (SSD)](https://arxiv.org/abs/1801.04381) network intended to perform object detection. The difference between this model and the `mobilenet-ssd` is that there the `mobilenet-ssd` can only detect face, the `ssd_mobilenet_v1_coco` model can detect objects.

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Detection     |
| GFLOPs            | 2.494         |
| MParams           | 6.807         |
| Source framework  | TensorFlow\*  |

## Accuracy

| Metric         | Value   |
| -------------- | ------- |
| coco_precision | 23.3212%|

## Input

### Original model

Image, name - `image_tensor`, shape - `1, 300, 300, 3`, format - `B, H, W, C`, where:

- `B` - batch size
- `H` - image height
- `W` - image width
- `C` - number of channels

Expected color order - `RGB`.

### Converted model

Image, name - `image_tensor`, shape - `1, 3, 300, 300`, format - `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order - `BGR`.

## Output

### Original model

1. Classifier, name - `detection_classes`, contains predicted bounding boxes classes in range [1, 91]. The model was trained on [Common Objects in Context (COCO)](https://cocodataset.org/#home) dataset version with 91 categories of object, 0 class is for background. Mapping to class names provided in `<omz_dir>/data/dataset_classes/coco_91cl_bkgr.txt` file.
2. Probability, name - `detection_scores`, contains probability of detected bounding boxes.
3. Detection box, name - `detection_boxes`, contains detection boxes coordinates in format `[y_min, x_min, y_max, x_max]`, where (`x_min`, `y_min`)  are coordinates top left corner, (`x_max`, `y_max`) are coordinates right bottom corner. Coordinates are rescaled to input image size.
4. Detections number, name - `num_detections`, contains the number of predicted detection boxes.

### Converted model

The array of summary detection information, name - `DetectionOutput`, shape - `1, 1, 100, 7` in the format `1, 1, N, 7`, where `N` is the number of detected
bounding boxes. For each detection, the description has the format:
[`image_id`, `label`, `conf`, `x_min`, `y_min`, `x_max`, `y_max`], where:

- `image_id` - ID of the image in the batch
- `label` - predicted class ID in range [1, 91], mapping to class names provided in `<omz_dir>/data/dataset_classes/coco_91cl_bkgr.txt` file.
- `conf` - confidence for the predicted class
- (`x_min`, `y_min`) - coordinates of the top left bounding box corner (coordinates stored in normalized format, in range [0, 1])
- (`x_max`, `y_max`) - coordinates of the bottom right bounding box corner  (coordinates stored in normalized format, in range [0, 1])

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

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/tensorflow/models/master/LICENSE).
A copy of the license is provided in [APACHE-2.0-TF-Models.txt](../licenses/APACHE-2.0-TF-Models.txt).
