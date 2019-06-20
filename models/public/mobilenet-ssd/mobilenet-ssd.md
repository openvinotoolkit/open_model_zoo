# mobilenet-ssd

## Use Case and High-Level Description

The `mobilenet-ssd` model is a Single-Shot multibox Detection (SSD) network intended to perform object detection. This model is implemented using the Caffe framework. For details about this model, check out the [repository](https://github.com/chuanqi305/MobileNet-SSD).

The model input is a blob that consists of a single image of "1x3x300x300" in BGR order, also like the `densenet-121` model. The BGR mean values need to be subtracted as follows: [127.5, 127.5, 127.5] before passing the image blob into the network. In addition, values must be scaled by 0.007843.

The model output is a typical vector containing the tracked object data, as previously described.

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Detection     |
| GFLOPs            | 2.316         |
| MParams           | 5.783         |
| Source framework  | Caffe         |

## Accuracy

## Performance

## Inputs

Name - `data`, shape - `1,3,300,300`, image format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`

## Outputs

Name: `detection_out`

## Legal Information

[https://raw.githubusercontent.com/tensorflow/models/master/LICENSE]()
