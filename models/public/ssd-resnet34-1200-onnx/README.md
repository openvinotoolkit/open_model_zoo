# ssd-resnet34-1200-onnx

## Use Case and High-Level Description

The `ssd-resnet-34-1200-onnx` model is a multiscale SSD based on ResNet-34 backbone network intended to perform object detection. The model has been trained from the Common Objects in Context (COCO) image dataset. This model is pre-trained in PyTorch\* framework and converted to ONNX\* format. For additional information refer to [repository](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection).

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Detection     |
| GFLOPs            | 433.411       |
| MParams           | 20.058        |
| Source framework  | PyTorch\*     |

## Accuracy

| Metric         | Value    |
| -------------- | -------- |
| coco_precision | 20.7198% |
| mAP            | 39.2752%	|

## Input

Note that original model expects image in `RGB` format, converted model - in `BGR` format.

### Original model

Image, shape - `1, 3, 1200, 1200`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `RGB`.

### Converted model

Image, shape - `1, 3, 1200, 1200`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

> **NOTE** output format changes after Model Optimizer conversion. To find detailed explanation of changes, go to [Model Optimizer development guide](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html)

### Original model

1. Classifier, name - `labels`, shape - `1, N`, contains predicted classes for each detected bounding box in [1, 81] range. The model was trained on [Common Objects in Context (COCO)](https://cocodataset.org/#home) dataset version with 80 categories of object,  0 class is for background. Mapping to class names provided in `<omz_dir>/data/dataset_classes/coco_80cl_bkgr.txt` file
2. Probability, name - `scores`, shape - `1, N`, contains confidence of each detected bounding boxes.
3. Detection boxes, name - `bboxes`, shape - `1, N, 4`, contains detection boxes coordinates in format `[y_min, x_min, y_max, x_max]`, where (`x_min`, `y_min`)  are coordinates top left corner, (`x_max`, `y_max`) are coordinates right bottom corner. Coordinates are rescaled to input image size.

### Converted model

1. Classifier, shape - `1, 200`, contains predicted class ID for each detected bounding box in [1, 81] range. The model was trained on [Common Objects in Context (COCO)](https://cocodataset.org/#home) dataset version with 80 categories of object, 0 class is for background. Mapping to class names provided in `<omz_dir>/data/dataset_classes/coco_80cl_bkgr.txt` file
2. Probability, shape - `1, 200`, contains confidence of each detected bounding boxes.
3. Detection boxes, shape - `1, 200, 4`, contains detection boxes coordinates in format `[y_min, x_min, y_max, x_max]`, where (`x_min`, `y_min`)  are coordinates top left corner, (`x_max`, `y_max`) are coordinates right bottom corner. Coordinates are in normalized format, in range [0, 1].

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
[Apache License, Version 2.0](https://raw.githubusercontent.com/mlcommons/inference/master/LICENSE.md).
A copy of the license is provided in [APACHE-2.0-MLPerf.txt](../licenses/APACHE-2.0-MLPerf.txt).
