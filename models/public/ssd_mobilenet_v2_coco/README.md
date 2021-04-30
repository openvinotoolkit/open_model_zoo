# ssd_mobilenet_v2_coco

## Use Case and High-Level Description

The `ssd_mobilenet_v2_coco` model is a [Single-Shot multibox Detection (SSD)](https://arxiv.org/abs/1801.04381) network intended to perform object detection. The model has been trained from the Common Objects in Context (COCO) image dataset.

The model input is a blob that consists of a single image of `1, 3, 300, 300` in `RGB` order.

The model output is a typical vector containing the tracked object data, as previously described. Note that the `class_id` data is now significant and should be used to determine the classification for any detected object.

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Detection     |
| GFLOPs            | 3.775         |
| MParams           | 16.818        |
| Source framework  | TensorFlow\*  |

## Accuracy

| Metric         | Value   |
| -------------- | ------- |
| coco_precision | 24.9452%|

## Input

Note that original model expects image in `RGB` format, converted model - in `BGR` format.

### Original model

Image, shape - `1, 300, 300, 3`, format is `B, H, W, C`, where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `RGB`.

### Converted model

Image, name - `image_tensor`, shape - `1, 300, 300, 3`, format is `B, H, W, C`, where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `BGR`.

## Output

> **NOTE** output format changes after Model Optimizer conversion. To find detailed explanation of changes, go to [Model Optimizer development guide](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html)

### Original model

1. Classifier, name - `detection_classes`, contains predicted bounding boxes classes in range [1, 91]. The model was trained on [Common Objects in Context (COCO)](https://cocodataset.org/#home) dataset version with 91 categories of object,  0 class is for background. Mapping to class names provided in `<omz_dir>/data/dataset_classes/coco_91cl_bkgr.txt` file.
2. Probability, name - `detection_scores`, contains probability of detected bounding boxes.
3. Detection box, name - `detection_boxes`, contains detection boxes coordinates in format `[y_min, x_min, y_max, x_max]`, where (`x_min`, `y_min`)  are coordinates top left corner, (`x_max`, `y_max`) are coordinates right bottom corner. Coordinates are rescaled to input image size.
4. Detections number, name - `num_detections`, contains the number of predicted detection boxes.

### Converted model

The array of summary detection information, name - `detection_out`,  shape - `1, 1, 100, 7` in the format `1, 1, N, 7`, where `N` is the number of detected bounding boxes. For each detection, the description has the format:
[`image_id`, `label`, `conf`, `x_min`, `y_min`, `x_max`, `y_max`], where:

- `image_id` - ID of the image in the batch
- `label` - predicted class ID in range [1, 91], mapping to class names provided in `<omz_dir>/data/dataset_classes/coco_91cl_bkgr.txt` file
- `conf` - confidence for the predicted class
- (`x_min`, `y_min`) - coordinates of the top left bounding box corner (coordinates are in normalized format, in range [0, 1])
- (`x_max`, `y_max`) - coordinates of the bottom right bounding box corner  (coordinates are in normalized format, in range [0, 1])

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
