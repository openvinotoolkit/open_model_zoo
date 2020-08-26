# face-detection-0105

## Use Case and High-Level Description

Face detector based on MobileNetV2 as a backbone with a
FCOS head for indoor and outdoor scenes shot by a front-facing camera.

## Example

![](./face-detection-0105.png)

## Specification

| Metric                                                        | Value                   |
|---------------------------------------------------------------|-------------------------|
| AP ([WIDER](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)) | 93.36%                  |
| GFlops                                                        | 2.853                   |
| MParams                                                       | 2.392                   |
| Source framework                                              | PyTorch*                |

Average Precision (AP) is defined as an area under the
[precision/recall](https://en.wikipedia.org/wiki/Precision_and_recall)
curve. All numbers were evaluated by taking into account only faces bigger than
64 x 64 pixels.

## Performance

## Inputs

Name: `input`, shape: [1x3x416x416] - An input image in the format [BxCxHxW],
where:

- B - batch size
- C - number of channels
- H - image height
- W - image width

Expected color order: BGR.

## Outputs

1. The `boxes` is a blob with the shape [N, 5], where N is the number of detected
   bounding boxes. For each detection, the description has the format
   [`x_min`, `y_min`, `x_max`, `y_max`, `conf`],
   where:
    - (`x_min`, `y_min`) - coordinates of the top left bounding box corner
    - (`x_max`, `y_max`) - coordinates of the bottom right bounding box corner
    - `conf` - confidence for the predicted class
2. The `labels` is a blob with the shape [N], where N is the number of detected
   bounding boxes. It contains `label` per each detected box.

## Legal Information
[*] Other names and brands may be claimed as the property of others.
