# ssd_mobilenet_v2_coco

## Use Case and High-Level Description

The `ssd_mobilenet_v2_coco` model is a [Single-Shot multibox Detection (SSD)](https://arxiv.org/pdf/1801.04381.pdf) network intended to perform object detection. The differnce bewteen this model and the `mobilenet-ssd` is that there the `mobilenet-ssd` can only detect face, the `ssd_mobilenet_v2_coco` model can detect objects as it has been trained from the Common Objects in COntext (COCO) image dataset. 

The model input is a blob that consists of a single image of "1x3x300x300" in BGR order.

The model output is a typical vector containing the tracked object data, as previously described. Note that the "class_id" data is now significant and should be used to determine the classification for any detected object.

## Example

## Specification

## Accuracy

## Performance

## Inputs

Name - `data`, shape - `1,300,300,3` image format is `B,H,W,C` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

## Outputs

1. Name: `detection_classes`
2. Name: `detection_scores`
3. Name: `detection_boxes`
4. Name: `num_detections`

## Legal Information

[https://raw.githubusercontent.com/tensorflow/models/master/LICENSE]()
