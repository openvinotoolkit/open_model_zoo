# faster_rcnn_inception_resnet_v2_atrous_coco

## Use Case and High-Level Description

Faster R-CNN with Inception Resnet v2 Atrous version. Used for object detection. For details see the [paper](https://arxiv.org/abs/1506.01497v3).

## Example

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Object detection                          |
| GFlops                          | 30.687                                    |
| MParams                         | 13.307                                    |
| Source framework                | TensorFlow\*                              |

## Accuracy

| Metric | Value |
| ------ | ----- |
| coco_precision | 36.76% |
| mAP| 52.41%|

## Performance

## Input

### Original Model

Image, name: `image_tensor`, shape: [1x600x1024x3], format: [BxHxWxC],
   where:

    - B - batch size
    - H - image height
    - W - image width
    - C - number of channels

   Expected color order: RGB.

### Converted Model

1. Image, name: `image_tensor`, shape: [1x3x600x1024], format: [BxCxHxW],
where:

   - B - batch size
   - C - number of channels
   - H - image height
   - W - image width

Expected color order: BGR.

2. Information of input image size, name: `image_info`, shape: [1x3], format: [BxC],
   where:

    - B - batch size
    - C - vector of three values in a format [H,W,S], where H is an image height, W is an image width, S is an image scale factor (usually 1).

## Output

### Original Model

1. Classifier, name: `detection_classes`. Contains predicted bounding boxes classes in a range [1, 91]. The model was trained on the Microsoft\* COCO dataset version with 90 categories of object.
2. Probability, name: `detection_scores`. Contains probability of detected bounding boxes.
3. Detection box, name: `detection_boxes`. Contains detection boxes coordinates in format `[y_min, x_min, y_max, x_max]`, where (`x_min`, `y_min`)  are coordinates of the top left corner, (`x_max`, `y_max`) are coordinates of the right bottom corner. Coordinates are rescaled to input image size.
4. Detections number, name: `num_detections`. Contains the number of predicted detection boxes.

### Converted Model

The array of summary detection information, name: `reshape_do_2d`, shape: [1, 1, N, 7], where N is the number of detected
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
[Apache License, Version 2.0](https://raw.githubusercontent.com/tensorflow/models/master/LICENSE).
A copy of the license is provided in [APACHE-2.0-TensorFlow.txt](../licenses/APACHE-2.0-TensorFlow.txt).
