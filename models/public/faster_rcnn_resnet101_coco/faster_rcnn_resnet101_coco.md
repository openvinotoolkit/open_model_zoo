# faster_rcnn_resnet101_coco

## Use Case and High-Level Description

Faster R-CNN Resnet-101 model. Used for object detection. For details see [paper](https://arxiv.org/pdf/1801.04381.pdf).

## Example

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Object detection                          |
| GFlops                          | 112.052                                   |
| MParams                         | 48.128                                    |
| Source framework                | Tensorflow                                |

## Performance

## Input

### Original model

1. Name: `image_tensor`, shape: [1x600x600x3] - An input image in the format [BxHxWxC],
   where:

    - B - batch size
    - H - image height
    - W - image width
    - C - number of channels

   Expected color order - RGB.

### Converted model

1. Name: `image_tensor`, shape: [1x3x600x600] - An input image in the format [BxCxHxW],
   where:

    - B - batch size
    - C - number of channels
    - H - image height
    - W - image width

   Expected color order - BGR.

1. Name: `image_info`, shape: [1x3], in format [BxC],
   where:

    - B - batch size
    - C - vector of 3 values in format [H,W,S], represents information of input image size, where H - image height, W - imahe width, S - image scale factor (usually 1)

## Output

### Original model

1. Name: `detection_classes` contains predicted bounding boxes classes in range [1, 91]. The model was trained on MS COCO dataset version with 90 categories of object.
2. Name: `detection_scores` probability of detected bounding boxes
3. Name: `detection_boxes` contains detection boxes coordinates in format `[y_min, x_min, y_max, x_max]` where (`x_min`, `y_min`)  is coordinates top left corner,  (`x_max`, `y_max`) is coordinates right bottom corner. Coordinates rescaled to input image size.
4. Name: `num_detections` contains the number of predicted detection boxes

### Converted model

1. Name: `reshape_do_2d`, shape: [1, 1, N, 7], where N is the number of detected
bounding boxes. For each detection, the description has the format:
[`image_id`, `label`, `conf`, `x_min`, `y_min`, `x_max`, `y_max`],
    where:

    - `image_id` - ID of the image in the batch
    - `label` - predicted class ID
    - `conf` - confidence for the predicted class
    - (`x_min`, `y_min`) - coordinates of the top left bounding box corner (coordinates stored in normalized format, in range [0, 1])
    - (`x_max`, `y_max`) - coordinates of the bottom right bounding box corner  (coordinates stored in normalized format, in range [0, 1])

## Legal Information
[https://raw.githubusercontent.com/tensorflow/models/master/LICENSE]()