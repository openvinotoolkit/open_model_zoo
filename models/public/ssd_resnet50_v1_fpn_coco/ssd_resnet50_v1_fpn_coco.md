# ssd_resnet50_v1_fpn_coco

## Use Case and High-Level Description

The `ssd_resnet50_v1_fpn_coco` model is a SSD FPN object detection architecture based on ResNet-50.
The model has been trained from the Common Objects in Context (COCO) image dataset.
For details see the [repository](https://github.com/tensorflow/models/blob/master/research/object_detection)
and [paper](https://arxiv.org/abs/1708.02002).

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Detection     |
| GFLOPs            | 178.6807      |
| MParams           | 56.9326       |
| Source framework  | TensorFlow\*  |

## Accuracy

| Metric | Value |
| ------ | ----- |
| coco_precision | 38.4557% |

## Performance

## Input

### Original model

Image, name - `image_tensor`, shape - `[1x640x640x3]`, format -`[BxHxWxC]` where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Expected color order -  `RGB`.

### Converted model

Image, name - `image_tensor`, shape - `[1x3x640x640]`, format is `[BxCxHxW]` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Expected color order - `BGR`.

## Output

> **NOTE** output format changes after Model Optimizer conversion. To find detailed explanation of changes, go to [Model Optimizer development guide](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html)

### Original model

1. Classifier, name - `detection_classes`, contains predicted bounding boxes classes in range [1, 91]. The model was trained on Microsoft\* COCO dataset version with 90 categories of object.
2. Probability, name - `detection_scores`, contains probability of detected bounding boxes.
3. Detection box, name - `detection_boxes`, contains detection boxes coordinates in format `[y_min, x_min, y_max, x_max]`, where (`x_min`, `y_min`)  are coordinates top left corner, (`x_max`, `y_max`) are coordinates right bottom corner. Coordinates are rescaled to input image size.
4. Detections number, name - `num_detections`, contains the number of predicted detection boxes.


### Converted model

The array of summary detection information, name - `detection_out`,  shape - `[1x1xNx7]`, where N is the number of detected bounding boxes. For each detection, the description has the format:
[`image_id`, `label`, `conf`, `x_min`, `y_min`, `x_max`, `y_max`], where:

- `image_id` - ID of the image in the batch
- `label` - predicted class ID
- `conf` - confidence for the predicted class
- (`x_min`, `y_min`) - coordinates of the top left bounding box corner (coordinates are in normalized format, in range [0, 1])
- (`x_max`, `y_max`) - coordinates of the bottom right bounding box corner  (coordinates are in normalized format, in range [0, 1])

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/tensorflow/models/master/LICENSE).
A copy of the license is provided in [APACHE-2.0-TensorFlow.txt](../licenses/APACHE-2.0-TensorFlow.txt).
