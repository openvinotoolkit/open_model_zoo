# face-detection-retail-0005

## Use Case and High-Level Description

Face detector based on MobileNetV2 as a backbone with a
single SSD head for indoor/outdoor scenes shot by a front-facing camera. The single SSD
head from 1/16 scale feature map has nine clustered prior boxes.

## Example

![](./face-detection-retail-0001.png)

## Specification

| Metric                                                        | Value                   |
|---------------------------------------------------------------|-------------------------|
| AP ([WIDER](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)) | 84.52%                  |
| GFlops                                                        | 0.982                   |
| MParams                                                       | 1.021                   |
| Source framework                                              | PyTorch*                |

Average Precision (AP) is defined as an area under the
[precision/recall](https://en.wikipedia.org/wiki/Precision_and_recall)
curve. All numbers were evaluated by taking into account only faces bigger than
60 x 60 pixels.

## Performance

## Inputs

Name: `input`, shape: [1x3x300x300] - An input image in the format [BxCxHxW],
where:

   - B - batch size
   - C - number of channels
   - H - image height
   - W - image width

Expected color order: BGR.

## Outputs

The net outputs blob with shape: [1, 1, N, 7], where N is the number of detected
bounding boxes. Each detection has the format
  [`image_id`, `label`, `conf`, `x_min`, `y_min`, `x_max`, `y_max`], where:
  - `image_id` - ID of the image in the batch
  - `label` - predicted class ID
  - `conf` - confidence for the predicted class
  - (`x_min`, `y_min`) - coordinates of the top left bounding box corner
  - (`x_max`, `y_max`) - coordinates of the bottom right bounding box corner.

## Legal Information
[*] Other names and brands may be claimed as the property of others.
