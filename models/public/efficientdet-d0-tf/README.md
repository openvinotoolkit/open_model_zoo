# efficientdet-d0-tf

## Use Case and High-Level Description

The `efficientdet-d0-tf` model is one of the [EfficientDet](https://arxiv.org/abs/1911.09070)
models  designed to perform object detection. This model was pre-trained in TensorFlow\*.
All the EfficientDet models have been pre-trained on the [Common Objects in Context (COCO)](https://cocodataset.org/#home) image database.
For details about this family of models, check out the Google AutoML [repository](https://github.com/google/automl/tree/master/efficientdet).

## Specification

| Metric            | Value           |
|-------------------|-----------------|
| Type              | Object detection|
| GFLOPs            |     2.54        |
| MParams           |     3.9         |
| Source framework  | TensorFlow\*    |

## Accuracy

| Metric                                                                | Converted model |
| --------------------------------------------------------------------- | --------------- |
| [COCO mAP (0.5:0.05:0.95)](https://cocodataset.org/#detection-eval)   | 31.95%          |

## Input

### Original Model

Image, name - `image_arrays`,  shape - `1, 512, 512, 3`, format is `B, H, W, C`, where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `RGB`.

### Converted Model

Image, name - `image_arrays/placeholder_port_0`,  shape - `1, 3, 512, 512`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original Model

The array of summary detection information, name: `detections`, shape: `1, 100, 7` in the format  `1, N, 7`, where `N` is the number of detected
bounding boxes. For each detection, the description has the format:
[`image_id`, `y_min`, `x_min`, `y_max`, `x_max`, `confidence`, `label`], where:

- `image_id` - ID of the image in the batch
- (`x_min`, `y_min`) - coordinates of the top left bounding box corner
- (`x_max`, `y_max`) - coordinates of the bottom right bounding box corner
- `confidence` - confidence for the predicted class
- `label` - predicted class ID, in range [1, 91], mapping to class names provided in `<omz_dir>/data/dataset_classes/coco_91cl.txt` file

### Converted Model

The array of summary detection information, name: `detections`, shape: `1, 1, 100, 7` in the format `1, 1, N, 7`, where `N` is the number of detected
bounding boxes. For each detection, the description has the format:
[`image_id`, `label`, `conf`, `x_min`, `y_min`, `x_max`, `y_max`], where:

- `image_id` - ID of the image in the batch
- `label` - predicted class ID, in range [0, 90], mapping to class names provided in `<omz_dir>/data/dataset_classes/coco_91cl.txt` file
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
[Apache License, Version 2.0](https://raw.githubusercontent.com/google/automl/master/LICENSE).
A copy of the license is provided in [APACHE-2.0-TF-AutoML.txt](../licenses/APACHE-2.0-TF-AutoML.txt).
