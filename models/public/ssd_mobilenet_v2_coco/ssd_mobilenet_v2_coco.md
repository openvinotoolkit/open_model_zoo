# ssd_mobilenet_v2_coco

## Use Case and High-Level Description

The `ssd_mobilenet_v2_coco` model is a [Single-Shot multibox Detection (SSD)](https://arxiv.org/pdf/1801.04381.pdf) network intended to perform object detection. The difference bewteen this model and the `mobilenet-ssd` is that there the `mobilenet-ssd` can only detect face, the `ssd_mobilenet_v2_coco` model can detect objects as it has been trained from the Common Objects in COntext (COCO) image dataset. 

The model input is a blob that consists of a single image of "1x3x300x300" in RGB order.

The model output is a typical vector containing the tracked object data, as previously described. Note that the "class_id" data is now significant and should be used to determine the classification for any detected object.

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Detection     |
| GFLOPs            | 3.775         |
| MParams           | 16.818        |
| Source framework  | Tensorflow    |

## Accuracy

## Performance

## Input

Note that original model expects image in `RGB` format, converted model - in `BGR` format.

### Original model

Image, shape - `1,300,300,3`, format is `B,H,W,C` where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `RGB`

### Converted model

Image, shape - `1,300,300,3`, format is `B,H,W,C` where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `BGR`

## Output

**ATTENTION!** After Model Optimizer's conversion original output format will be changed. Detailed explanation of changes after Model Optimizer's conversion you can find in [Model Optimizer development guide](http://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html)

### Original model 

1. Name: `detection_classes` contains predicted bounding boxes classes in range [1, 91]. The model was trained on MS COCO dataset version with 90 categories of object.
2. Name: `detection_scores` probability of detected bounding boxes
3. Name: `detection_boxes` contains detection boxes coordinates in format `[y_min, x_min, y_max, x_max]` where (`x_min`, `y_min`)  is coordinates top left corner,  (`x_max`, `y_max`) is coordinates right bottom corner. Coordinates rescaled to input image size.
4. Name: `num_detections` contains the number of predicted detection boxes


### Model in IR format

The net outputs a blob with shape: [1, 1, N, 7], where N is the number of detected
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
