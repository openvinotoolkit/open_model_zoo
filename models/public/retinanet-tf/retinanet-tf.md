# retinanet-tf

## Use Case and High-Level Description

RetinaNet is the dense object detection model with ResNet50 backbone.
For details, see [paper](https://arxiv.org/abs/1708.02002),
[repository](https://github.com/fizyr/keras-retinanet).

## Example

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Object detection                          |
| GFlops                          | 238.9469                                  |
| MParams                         | 64.9706                                   |
| Source framework                | TensorFlow\*                              |

## Performance

## Input

### Original Model

Image, name: `input_1`, shape: [1x1333x1333x3], format: [BxHxWxC],
   where:

    - B - batch size
    - H - image height
    - W - image width
    - C - number of channels

   Expected color order: RGB.
Mean values: [103.939, 116.779, 123.68]

### Converted Model

Image, name: `input_1`, shape: [1x3x1333x1333], format: [BxCxHxW],
   where:

    - B - batch size
    - C - number of channels
    - H - image height
    - W - image width

   Expected color order: BGR.

## Output

### Original Model

1. Classifier, name: `filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3`. Contains predicted bounding boxes classes in a range [1, 80]. The model was trained on the Microsoft\* COCO dataset version with 80 categories of objects.
2. Probability, name: `filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3`. Contains probability of detected bounding boxes.
3. Detection box, name: `filtered_detections/map/TensorArrayStack/TensorArrayGatherV3`. Contains detection boxes coordinates in a format `[y_min, x_min, y_max, x_max]`, where (`x_min`, `y_min`)  are coordinates of the top left corner, (`x_max`, `y_max`) are coordinates of the right bottom corner. Coordinates are rescaled to input image size.

### Converted Model

The array of summary detection information, name - `DetectionOutput`, shape - [1, 1, N, 7], where N is the number of detected
bounding boxes. For each detection, the description has the format:
[`image_id`, `label`, `conf`, `x_min`, `y_min`, `x_max`, `y_max`],
    where:

    - `image_id` - ID of the image in the batch
    - `label` - predicted class ID
    - `conf` - confidence for the predicted class
    - (`x_min`, `y_min`) - coordinates of the top left bounding box corner (coordinates stored in normalized format, in range [0, 1])
    - (`x_max`, `y_max`) - coordinates of the bottom right bounding box corner  (coordinates stored in normalized format, in range [0, 1])

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/fizyr/keras-retinanet/master/LICENSE).
A copy of the license is provided in [APACHE-2.0.txt](../licenses/APACHE-2.0.txt).
