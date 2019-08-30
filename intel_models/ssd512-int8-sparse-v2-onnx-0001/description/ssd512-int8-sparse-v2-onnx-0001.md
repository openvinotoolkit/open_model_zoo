# ssd512-int8-sparse-v2-onnx-0001

## Use Case and High-Level Description

This is an SSD model with a VGG16 backbone that is designed to perform object detection. The model has been pretrained on the VOC image database (VOC07+12 dataset) and then symmetrically quantized to INT8 fixed-point precision and pruned to 77.8% sparsity rate using Neural Network Compression Framework (NNCF).

The model input is a blob that consists of a single image of "1x3x512x512" in RGB order.

The model output for `ssd512-int8-sparse-v2-onnx-0001` is the usual object detection output for the 21 different classes present in the VOC database.

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Detection |
| GFLOPs            | 180.611 |
| MParams           | 27.189 |
| Source framework  | PyTorch    |

## Accuracy

The quality metric (mAP) calculated on VOC07+12 validation dataset has the value of 77.2%.

| Metric                    | Value         |
|---------------------------|---------------|
| mAP |         77.2% |

## Input

Image, shape - `1,3,512,512`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `RGB`

## Output

The net outputs a blob with shape: [1, 1, N, 7], where N is the number of detected bounding boxes (N=200 for this model). For each detection, the description has the format: [`image_id`, `label`, `conf`, `x_min`, `y_min`, `x_max`, `y_max`], where:

- `image_id` - ID of the image in the batch
- `label` - predicted class ID
- `conf` - confidence for the predicted class
- (`x_min`, `y_min`) - coordinates of the top left bounding box corner
- (`x_max`, `y_max`) - coordinates of the bottom right bounding box corner
