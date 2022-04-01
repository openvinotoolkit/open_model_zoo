# face-detection-retail-0004

## Use Case and High-Level Description

Face detector based on SqueezeNet light (half-channels) as a backbone with a
single SSD for indoor/outdoor scenes shot by a front-facing camera. The backbone
consists of fire modules to reduce the number of computations. The single SSD
head from 1/16 scale feature map has nine clustered prior boxes.

## Example

![](./assets/face-detection-retail-0004.png)

## Specification

| Metric                                                        | Value                   |
|---------------------------------------------------------------|-------------------------|
| AP ([WIDER](http://shuoyang1213.me/WIDERFACE/))               | 83.00%                  |
| GFlops                                                        | 1.067                   |
| MParams                                                       | 0.588                   |
| Source framework                                              | Caffe\*                 |

Average Precision (AP) is defined as an area under the
[precision/recall](https://en.wikipedia.org/wiki/Precision_and_recall)
curve. All numbers were evaluated by taking into account only faces bigger than
60 x 60 pixels.

## Inputs

Image, name: `data`, shape: `1, 3, 300, 300` in the format `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: `BGR`.

## Outputs

The net outputs blob with shape: `1, 1, 200, 7` in the format `1, 1, N, 7`, where `N` is the number of detected
bounding boxes. Each detection has the format [`image_id`, `label`, `conf`, `x_min`, `y_min`, `x_max`, `y_max`], where:

- `image_id` - ID of the image in the batch
- `label` - predicted class ID (1 - face)
- `conf` - confidence for the predicted class
- (`x_min`, `y_min`) - coordinates of the top left bounding box corner
- (`x_max`, `y_max`) - coordinates of the bottom right bounding box corner

## Demo usage

The model can be used in the following demos provided by the Open Model Zoo to show its capabilities:

* [Face Recognition Python\* Demo](../../../demos/face_recognition_demo/python/README.md)
* [Gaze Estimation Demo](../../../demos/gaze_estimation_demo/cpp/README.md)
* [G-API Gaze Estimation Demo](../../../demos/gaze_estimation_demo/cpp_gapi/README.md)
* [Interactive Face Detection C++ Demo](../../../demos/interactive_face_detection_demo/cpp/README.md)
* [G-API Interactive Face Detection Demo](../../../demos/interactive_face_detection_demo/cpp_gapi/README.md)
* [Multi-Channel Face Detection C++ Demo](../../../demos/multi_channel_face_detection_demo/cpp/README.md)
* [Object Detection C++ Demo](../../../demos/object_detection_demo/cpp/README.md)
* [Object Detection Python\* Demo](../../../demos/object_detection_demo/python/README.md)

## Legal Information
[*] Other names and brands may be claimed as the property of others.
