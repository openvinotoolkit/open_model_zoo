# mobilenet-ssd

## Use Case and High-Level Description

The `mobilenet-ssd` model is a Single-Shot multibox Detection (SSD) network intended to perform object detection. This model is implemented using the Caffe\* framework. For details about this model, check out the [repository](https://github.com/chuanqi305/MobileNet-SSD).

The model input is a blob that consists of a single image of 1x3x300x300 in BGR order, also like the `densenet-121` model. The BGR mean values need to be subtracted as follows: [127.5, 127.5, 127.5] before passing the image blob into the network. In addition, values must be divided by 0.007843.

The model output is a typical vector containing the tracked object data, as previously described.

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Detection     |
| GFLOPs            | 2.316         |
| MParams           | 5.783         |
| Source framework  | Caffe\*         |

## Accuracy

See [https://github.com/chuanqi305/MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD).

## Performance

## Input

### Original model

Image, name - `prob`,  shape - `1,3,300,300`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.
Mean values - [127.5, 127.5, 127.5], scale value - 127.50223128904757.

### Converted model

Image, name - `prob`,  shape - `1,3,300,300`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`

## Output

### Original model 

The array of detection summary info, name - `detection_out`,  shape - `1, 1, N, 7`, where N is the number of detected bounding boxes. For each detection, the description has the format:
[`image_id`, `label`, `conf`, `x_min`, `y_min`, `x_max`, `y_max`], where:

- `image_id` - ID of the image in the batch
- `label` - predicted class ID
- `conf` - confidence for the predicted class
- (`x_min`, `y_min`) - coordinates of the top left bounding box corner (coordinates are in normalized format, in range [0, 1])
- (`x_max`, `y_max`) - coordinates of the bottom right bounding box corner  (coordinates are in normalized format, in range [0, 1])

### Converted model

The array of detection summary info, name - `detection_out`,  shape - `1, 1, N, 7`, where N is the number of detected bounding boxes. For each detection, the description has the format:
[`image_id`, `label`, `conf`, `x_min`, `y_min`, `x_max`, `y_max`], where:

- `image_id` - ID of the image in the batch
- `label` - predicted class ID
- `conf` - confidence for the predicted class
- (`x_min`, `y_min`) - coordinates of the top left bounding box corner (coordinates are in normalized format, in range [0, 1])
- (`x_max`, `y_max`) - coordinates of the bottom right bounding box corner  (coordinates are in normalized format, in range [0, 1])

## Legal Information

[https://raw.githubusercontent.com/tensorflow/models/master/LICENSE]()
