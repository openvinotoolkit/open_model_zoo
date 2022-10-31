# face-detection-adas-0001

## Use Case and High-Level Description

Face detector for driver monitoring and similar scenarios. The network features
a default MobileNet backbone that includes depth-wise convolutions to reduce the
amount of computation for the 3x3 convolution block.

## Example

![](./assets/face-detection-adas-0001.png)

## Specification

| Metric                          | Value                 |
|---------------------------------|-----------------------|
| AP (head height >10px)          | 37.4%                 |
| AP (head height >32px)          | 84.8%                 |
| AP (head height >64px)          | 93.1%                 |
| AP (head height >100px)         | 94.1%                 |
| Min head size                   | 90x90 pixels on 1080p |
| GFlops                          | 2.835                 |
| MParams                         | 1.053                 |
| Source framework                | Caffe\*               |

Average Precision (AP) is defined as an area under the
[precision/recall](https://en.wikipedia.org/wiki/Precision_and_recall)
curve. Numbers are on
[Wider Face](http://shuoyang1213.me/WIDERFACE/) validation subset.

## Inputs

Image, name: `data`, shape: `1, 3, 384, 672` in the format `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order is `BGR`.

## Outputs

The net outputs blob with shape: `1, 1, 200, 7` in the format `1, 1, N, 7`, where `N` is the number of detected
bounding boxes. The results are sorted by confidence in decreasing order. Each detection has the format
[`image_id`, `label`, `conf`, `x_min`, `y_min`, `x_max`, `y_max`], where:

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
* [Smart Classroom C++ Demo](../../../demos/smart_classroom_demo/cpp/README.md)
* [Smart Classroom C++ G-API Demo](../../../demos/smart_classroom_demo/cpp_gapi/README.md)

## Legal Information
[*] Other names and brands may be claimed as the property of others.
