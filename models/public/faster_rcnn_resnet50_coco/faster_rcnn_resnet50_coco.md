# faster_rcnn_resnet50_coco

## Use Case and High-Level Description

Faster R-CNN Resnet-50 model. Used for object detection. For details see [paper](https://arxiv.org/pdf/1801.04381.pdf).

## Example

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Object detection                          |
| GFlops                          | 57.203                                    |
| MParams                         | 29.162                                    |
| Source framework                | Tensorflow\*                              |

## Performance

## Input

### Original model

Image, name: `image_tensor`, shape: [1x600x600x3], format [BxHxWxC],
   where:

    - B - batch size
    - H - image height
    - W - image width
    - C - number of channels

   Expected color order - RGB.

### Converted model

1. Image, name: `image_tensor`, shape: [1x3x600x600], format [BxCxHxW],
   where:

    - B - batch size
    - C - number of channels
    - H - image height
    - W - image width

   Expected color order - BGR.

2. Information of input image size, name: `image_info`, shape: [1x3], in format [BxC],
   where:

    - B - batch size
    - C - vector of 3 values in format [H,W,S], where H - image height, W - imahe width, S - image scale factor (usually 1)

## Output

### Original model

1. Classifier, name - `detection_classes`, contains predicted bounding boxes classes in range [1, 91]. The model was trained on Microsoft\* COCO dataset version with 90 categories of object.
2. Probability, name - `detection_scores`, contains probability of detected bounding boxes.
3. Detection box, name - `detection_boxes`, contains detection boxes coordinates in format `[y_min, x_min, y_max, x_max]`, where (`x_min`, `y_min`)  are coordinates top left corner, (`x_max`, `y_max`) are coordinates right bottom corner. Coordinates are rescaled to input image size.
4. Detections number, name - `num_detections`, contains the number of predicted detection boxes.

### Converted model

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

[https://raw.githubusercontent.com/tensorflow/models/master/LICENSE]()