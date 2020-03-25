# Overview of OpenVINO&trade; Toolkit Pre-Trained Models

OpenVINO&trade; toolkit provides a set of pre-trained models
that you can use for learning and demo purposes or for developing deep learning
software. Most recent version is available in the [repo on Github](https://github.com/opencv/open_model_zoo).

The models can be downloaded via Model Downloader
(`<OPENVINO_INSTALL_DIR>/deployment_tools/open_model_zoo/tools/downloader`).
They can also be downloaded manually from [01.org](https://download.01.org/opencv).

## Object Detection Models

Several detection models can be used to detect a set of the most popular
objects - for example, faces, people, vehicles. Most of the networks are
SSD-based and provide reasonable accuracy/performance trade-offs. Networks that
detect the same types of objects (for example, `face-detection-adas-0001` and
`face-detection-retail-0004`) provide a choice for higher accuracy/wider
applicability at the cost of slower performance, so you can expect a "bigger"
network to detect objects of the same type better.

| Model Name                                                                                                                                                                          | Complexity (GFLOPs)  | Size (Mp)  | Face  | Person  | Vehicle  | Bike  | License plate  | Product |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |--------------------- |----------- |------ |-------- |--------- |------ |--------------- |-------- |
| [faster-rcnn-resnet101-coco-sparse-60-0001](./faster-rcnn-resnet101-coco-sparse-60-0001/description/faster-rcnn-resnet101-coco-sparse-60-0001.md)                                   | 364.21               | 52.79      |       | X       | X        | X     |                |         |
| [face-detection-adas-0001](./face-detection-adas-0001/description/face-detection-adas-0001.md)                                                                                      | 2.835                | 1.053      | X     |         |          |       |                |         |
| [face-detection-adas-binary-0001](./face-detection-adas-binary-0001/description/face-detection-adas-binary-0001.md)                                                                 | 0.819                | 1.053      | X     |         |          |       |                |         |
| [face-detection-retail-0004](./face-detection-retail-0004/description/face-detection-retail-0004.md)                                                                                | 1.067                | 0.588      | X     |         |          |       |                |         |
| [face-detection-retail-0005](./face-detection-retail-0005/description/face-detection-retail-0005.md)                                                                                | 0.982                | 1.021      | X     |         |          |       |                |         |
| [face-detection-0100](./face-detection-0100/description/face-detection-0100.md)                                                                                                     | 0.785                | 1.828      | X     |         |          |       |                |         |
| [face-detection-0102](./face-detection-0102/description/face-detection-0102.md)                                                                                                     | 1.767                | 1.842      | X     |         |          |       |                |         |
| [face-detection-0104](./face-detection-0104/description/face-detection-0104.md)                                                                                                     | 2.405                | 1.851      | X     |         |          |       |                |         |
| [face-detection-0105](./face-detection-0105/description/face-detection-0105.md)                                                                                                     | 2.853                | 2.392      | X     |         |          |       |                |         |
| [face-detection-0106](./face-detection-0106/description/face-detection-0106.md)                                                                                                     | 339.597              | 69.920     | X     |         |          |       |                |         |
| [person-detection-retail-0002](./person-detection-retail-0002/description/person-detection-retail-0002.md)                                                                          | 12.427               | 3.244      |       | X       |          |       |                |         |
| [person-detection-retail-0013](./person-detection-retail-0013/description/person-detection-retail-0013.md)                                                                          | 2.300                | 0.723      |       | X       |          |       |                |         |
| [person-detection-action-recognition-0005](./person-detection-action-recognition-0005/description/person-detection-action-recognition-0005.md)                                      | 7.140                | 1.951      |       | X       |          |       |                |         |
| [person-detection-action-recognition-0006](./person-detection-action-recognition-0006/description/person-detection-action-recognition-0006.md)                                      | 8.225                | 2.001      |       | X       |          |       |                |         |
| [person-detection-action-recognition-teacher-0002](./person-detection-action-recognition-teacher-0002/description/person-detection-action-recognition-teacher-0002.md)              | 7.140                | 1.951      |       | X       |          |       |                |         |
| [person-detection-raisinghand-recognition-0001](./person-detection-raisinghand-recognition-0001/description/person-detection-raisinghand-recognition-0001.md)                       | 7.138                | 1.951      |       | X       |          |       |                |         |
| [pedestrian-detection-adas-0002](./pedestrian-detection-adas-0002/description/pedestrian-detection-adas-0002.md)                                                                    | 2.836                | 1.165      |       | X       |          |       |                |         |
| [pedestrian-detection-adas-binary-0001](./pedestrian-detection-adas-binary-0001/description/pedestrian-detection-adas-binary-0001.md)                                               | 0.945                | 1.165      |       | X       |          |       |                |         |
| [pedestrian-and-vehicle-detector-adas-0001](./pedestrian-and-vehicle-detector-adas-0001/description/pedestrian-and-vehicle-detector-adas-0001.md)                                   | 3.974                | 1.650      |       | X       | X        |       |                |         |
| [vehicle-detection-adas-0002](./vehicle-detection-adas-0002/description/vehicle-detection-adas-0002.md)                                                                             | 2.798                | 1.079      |       |         | X        |       |                |         |
| [vehicle-detection-adas-binary-0001](./vehicle-detection-adas-binary-0001/description/vehicle-detection-adas-binary-0001.md)                                                        | 0.942                | 1.079      |       |         | X        |       |                |         |
| [person-vehicle-bike-detection-crossroad-0078](./person-vehicle-bike-detection-crossroad-0078/description/person-vehicle-bike-detection-crossroad-0078.md)                          | 3.964                | 1.178      |       | X       | X        | X     |                |         |
| [person-vehicle-bike-detection-crossroad-1016](./person-vehicle-bike-detection-crossroad-1016/description/person-vehicle-bike-detection-crossroad-1016.md)                          | 3.560                | 2.887      |       | X       | X        | X     |                |         |
| [vehicle-license-plate-detection-barrier-0106](./vehicle-license-plate-detection-barrier-0106/description/vehicle-license-plate-detection-barrier-0106.md)                          | 0.349                | 0.634      |       |         | X        |       | X              |         |
| [product-detection-0001](./product-detection-0001/description/product-detection-0001.md)                                                                                            | 3.598                | 3.212      |       |         |          |       |                | X       |
| [person-detection-asl-0001](./person-detection-asl-0001/description/person-detection-asl-0001.md)                                                                                   | 0.986                | 1.338      |       |      X  |          |       |                |         |
| [yolo-v2-ava-0001](./yolo-v2-ava-0001/description/yolo-v2-ava-0001.md)                                                                                                              | 29.38                | 48.29      |       |      X  | X        | X     |                |         |
| [yolo-v2-ava-sparse-35-0001](./yolo-v2-ava-sparse-35-0001/description/yolo-v2-ava-sparse-35-0001.md)                                                                                | 29.38                | 48.29      |       |      X  | X        | X     |                |         |
| [yolo-v2-ava-sparse-70-0001](./yolo-v2-ava-sparse-70-0001/description/yolo-v2-ava-sparse-70-0001.md)                                                                                | 29.38                | 48.29      |       |      X  | X        | X     |                |         |
| [yolo-v2-tiny-ava-0001](./yolo-v2-tiny-ava-0001/description/yolo-v2-tiny-ava-0001.md)                                                                                               | 6.975                | 15.12      |       |      X  | X        | X     |                |         |
| [yolo-v2-tiny-ava-sparse-30-0001](./yolo-v2-tiny-ava-sparse-30-0001/description/yolo-v2-tiny-ava-sparse-30-0001.md)                                                                 | 6.975                | 15.12      |       |      X  | X        | X     |                |         |
| [yolo-v2-tiny-ava-sparse-60-0001](./yolo-v2-tiny-ava-sparse-60-0001/description/yolo-v2-tiny-ava-sparse-60-0001.md)                                                                 | 6.975                | 15.12      |       |      X  | X        | X     |                |         |


## Object Recognition Models

Object recognition models are used for classification, regression, and character
recognition. Use these networks after a respective detector (for example,
Age/Gender recognition after Face Detection).

| Model Name                                                                                                                                                  | Complexity (GFLOPs)  | Size (Mp)  |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------- |----------- |
| [age-gender-recognition-retail-0013](./age-gender-recognition-retail-0013/description/age-gender-recognition-retail-0013.md)                                | 0.094                | 2.138      |
| [head-pose-estimation-adas-0001](./head-pose-estimation-adas-0001/description/head-pose-estimation-adas-0001.md)                                            | 0.105                | 1.911      |
| [license-plate-recognition-barrier-0001](./license-plate-recognition-barrier-0001/description/license-plate-recognition-barrier-0001.md)                    | 0.328                | 1.218      |
| [vehicle-attributes-recognition-barrier-0039](./vehicle-attributes-recognition-barrier-0039/description/vehicle-attributes-recognition-barrier-0039.md)     | 0.126                | 0.626      |
| [emotions-recognition-retail-0003](./emotions-recognition-retail-0003/description/emotions-recognition-retail-0003.md)                                      | 0.126                | 2.483      |
| [landmarks-regression-retail-0009](./landmarks-regression-retail-0009/description/landmarks-regression-retail-0009.md)                                      | 0.021                | 0.191      |
| [facial-landmarks-35-adas-0002](./facial-landmarks-35-adas-0002/description/facial-landmarks-35-adas-0002.md)                                               | 0.042                | 4.595      |
| [person-attributes-recognition-crossroad-0230](./person-attributes-recognition-crossroad-0230/description/person-attributes-recognition-crossroad-0230.md)  | 0.174                | 0.735      |
| [gaze-estimation-adas-0002](./gaze-estimation-adas-0002/description/gaze-estimation-adas-0002.md)                                                           | 0.139                | 1.882      |

## Reidentification Models

Precise tracking of objects in a video is a common application of Computer
Vision (for example, for people counting). It is often complicated by a set of
events that can be described as a "relatively long absence of an object". For
example, it can be caused by occlusion or out-of-frame movement. In such cases,
it is better to recognize the object as "seen before" regardless of its current
position in an image or the amount of time passed since last known position.

The following networks can be used in such scenarios. They take an image of a
person and evaluate an embedding - a vector in high-dimensional space that
represents an appearance of this person. This vector can be used for further
evaluation: images that correspond to the same person will have embedding
vectors that are "close" by L2 metric (Euclidean distance).

There are multiple models that provide various trade-offs between performance
and accuracy (expect a bigger model to perform better).

| Model Name                                                                                                                        | Complexity (GFLOPs)  | Size (Mp)  | Rank-1 on Market-1501 |
|-----------------------------------------------------------------------------------------------------------------------------------|--------------------- |----------- |------------------ |
| [person-reidentification-retail-0031](./person-reidentification-retail-0031/description/person-reidentification-retail-0031.md)   | 0.028                | 0.280      | 92.11%            |
| [person-reidentification-retail-0248](./person-reidentification-retail-0248/description/person-reidentification-retail-0248.md)   | 0.174                | 0.183      | 84.3%             |
| [person-reidentification-retail-0249](./person-reidentification-retail-0249/description/person-reidentification-retail-0249.md)   | 0.564                | 0.597      | 92.9%             |
| [person-reidentification-retail-0300](./person-reidentification-retail-0300/description/person-reidentification-retail-0300.md)   | 3.521                | 5.289      | 96.3%             |

| Model Name                                                                                                                        | Complexity (GFLOPs)  | Size (Mp)  | Pairwise accuracy on LFW |
|-----------------------------------------------------------------------------------------------------------------------------------|--------------------- |----------- |------------------ |
| [face-reidentification-retail-0095](./face-reidentification-retail-0095/description/face-reidentification-retail-0095.md)         | 0.588                | 1.107      | 99.33%            |


## Semantic Segmentation Models

Semantic segmentation is an extension of object detection problem. Instead of
returning bounding boxes, semantic segmentation models return a "painted"
version of the input image, where the "color" of each pixel represents a certain
class. These networks are much bigger than respective object detection networks,
but they provide a better (pixel-level) localization of objects and they can
detect areas with complex shape (for example, free space on the road).

| Model Name                                                                                                                                     | Complexity (GFLOPs)  | Size (Mp)  |
|------------------------------------------------------------------------------------------------------------------------------------------------|--------------------- |----------- |
| [road-segmentation-adas-0001](./road-segmentation-adas-0001/description/road-segmentation-adas-0001.md)                                        | 4.770                | 0.184      |
| [semantic-segmentation-adas-0001](./semantic-segmentation-adas-0001/description/semantic-segmentation-adas-0001.md)                            | 58.572               | 6.686      |
| [unet-camvid-onnx-0001](./unet-camvid-onnx-0001/description/unet-camvid-onnx-0001.md)                                                          | 260.1                | 31.03      |
| [icnet-camvid-ava-0001](./icnet-camvid-ava-0001/description/icnet-camvid-ava-0001.md)                                                   | 151.82                | 25.45       |
| [icnet-camvid-ava-sparse-30-0001](./icnet-camvid-ava-sparse-30-0001/description/icnet-camvid-ava-sparse-30-0001.md)                     | 151.82                | 25.45       |
| [icnet-camvid-ava-sparse-60-0001](./icnet-camvid-ava-sparse-60-0001/description/icnet-camvid-ava-sparse-60-0001.md)                     | 151.82                | 25.45       |

## Instance Segmentation Models

Instance segmentation is an extension of object detection and semantic
segmentation problems. Instead of predicting a bounding box around each object
instance instance segmentation model outputs pixel-wise masks for all instances.

| Model Name                                                                                                                                     | Complexity (GFLOPs)  | Size (Mp)  |
|------------------------------------------------------------------------------------------------------------------------------------------------|--------------------- |----------- |
| [instance-segmentation-security-1025](./instance-segmentation-security-1025/description/instance-segmentation-security-1025.md)                | 30.146               | 26.69      |
| [instance-segmentation-security-0050](./instance-segmentation-security-0050/description/instance-segmentation-security-0050.md)                | 46.602               | 30.448     |
| [instance-segmentation-security-0083](./instance-segmentation-security-0083/description/instance-segmentation-security-0083.md)                | 365.626              | 143.444    |
| [instance-segmentation-security-0010](./instance-segmentation-security-0010/description/instance-segmentation-security-0010.md)                | 899.568              | 174.568    |


## Human Pose Estimation Models

Human pose estimation task is to predict a pose: body skeleton, which consists
of keypoints and connections between them, for every person in an input image or
video.  Keypoints are body joints, i.e. ears, eyes, nose, shoulders, knees, etc.
There are two major groups of such metods: top-down and bottom-up.  The first
detects persons in a given frame, crops or rescales detections, then runs pose
estimation network for every detection. These methods are very accurate. The
second finds all keypoints in a given frame, then groups them by person
instances, thus faster than previous, because network runs once.

| Model Name                                                                                                                 | Complexity (GFLOPs)  | Size (Mp)       |
|----------------------------------------------------------------------------------------------------------------------------|--------------------- |---------------- |
| [human-pose-estimation-0001](./human-pose-estimation-0001/description/human-pose-estimation-0001.md)                       | 15.435               | 4.099      |

## Image Processing

Deep Learning models find their application in various image processing tasks to
increase the quality of the output.

| Model Name                                                                                                                                  | Complexity (GFLOPs)  | Size (Mp)  |
|---------------------------------------------------------------------------------------------------------------------------------------------|--------------------- |----------- |
| [single-image-super-resolution-1032](./single-image-super-resolution-1032/description/single-image-super-resolution-1032.md)                | 11.654               | 0.030      |
| [single-image-super-resolution-1033](./single-image-super-resolution-1033/description/single-image-super-resolution-1033.md)                | 16.062               | 0.030      |
| [text-image-super-resolution-0001](./text-image-super-resolution-0001/description/text-image-super-resolution-0001.md)                      | 1.379                | 0.003      |

## Text Detection

Deep Learning models for text detection in various applications.

| Model Name                                                                                                                                     | Complexity (GFLOPs)  | Size (Mp)  |
|------------------------------------------------------------------------------------------------------------------------------------------------|--------------------- |----------- |
| [text-detection-0003](./text-detection-0003/description/text-detection-0003.md)                                                                | 51.256               | 6.747      |
| [text-detection-0004](./text-detection-0004/description/text-detection-0004.md)                                                                | 23.305               | 4.328      |

## Text Recognition

Deep Learning models for text recognition in various applications.

| Model Name                                                                                                                                     | Complexity (GFLOPs)  | Size (Mp)  |
|------------------------------------------------------------------------------------------------------------------------------------------------|--------------------- |----------- |
| [text-recognition-0012](./text-recognition-0012/description/text-recognition-0012.md)                                                          | 1.485                | 5.568      |
| [handwritten-score-recognition-0003](./handwritten-score-recognition-0003/description/handwritten-score-recognition-0003.md)                   | 0.792                | 5.555      |
| [handwritten-japanese-recognition-0001](./handwritten-japanese-recognition-0001/description/handwritten-japanese-recognition-0001.md)          | 117.136              | 15.31      |

## Text Spotting

Deep Learning models for text spotting (simultaneous detection and recognition).

| Model Name                                                                                                                                     | Complexity (GFLOPs)  | Size (Mp)  |
|------------------------------------------------------------------------------------------------------------------------------------------------|--------------------- |----------- |
| [text-spotting-0002-detector](./text-spotting-0002-detector/description/text-spotting-0002-detector.md)                                        | 185.169             | 26.497    |
| [text-spotting-0002-recognizer-encoder](./text-spotting-0002-recognizer-encoder/description/text-spotting-0002-recognizer-encoder.md)          | 2.082                | 1.328      |
| [text-spotting-0002-recognizer-decoder](./text-spotting-0002-recognizer-decoder/description/text-spotting-0002-recognizer-decoder.md)          | 0.002                | 0.273      |

## Action Recognition Models

Action Recognition models predict action that is being performed on a short video clip
(tensor formed by stacking sampled frames from input video). Some models (for example `driver-action-recognition-adas-0002` may use precomputed high-level spatial
or spatio-temporal) features (embeddings) from individual clip fragments and then aggregate them in a temporal model
to predict a vector with classification scores. Models that compute embeddings are called *encoder*, while models
that predict an actual labels are called *decoder*.

| Model Name                                                                                                                                              | Complexity (GFLOPs)  | Size (Mp)  |
|---------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------- |----------- |
| [driver-action-recognition-adas-0002-encoder](./driver-action-recognition-adas-0002-encoder/description/driver-action-recognition-adas-0002-encoder.md) | 0.676                | 2.863      |
| [driver-action-recognition-adas-0002-decoder](./driver-action-recognition-adas-0002-decoder/description/driver-action-recognition-adas-0002-decoder.md) | 0.147                | 4.205      |
| [action-recognition-0001-encoder](./action-recognition-0001-encoder/description/action-recognition-0001-encoder.md)                                     | 7.340                | 21.276     |
| [action-recognition-0001-decoder](./action-recognition-0001-decoder/description/action-recognition-0001-decoder.md)                                     | 0.147                | 4.405      |
| [asl-recognition-0004](./asl-recognition-0004/description/asl-recognition-0004.md)                                                                      | 6.660                | 4.133      |

## Image Retrieval

Deep Learning models for image retrieval (ranking 'gallery' images according to their similarity to some 'probe' image).

| Model Name                                                                                                                                     | Complexity (GFLOPs)  | Size (Mp)  |
|------------------------------------------------------------------------------------------------------------------------------------------------|--------------------- |----------- |
| [image-retrieval-0001](./image-retrieval-0001/description/image-retrieval-0001.md)                                                          | 0.613                | 2.535      |

## Compressed models

Deep Learning compressed models

| Model Name                                                                                                                                     | Complexity (GFLOPs)  | Size (Mp)  |
|------------------------------------------------------------------------------------------------------------------------------------------------|--------------------- |----------- |
| [resnet50-binary-0001](./resnet50-binary-0001/description/resnet50-binary-0001.md)                                                             | 1.002                | 7.446      |
| [resnet18-xnor-binary-onnx-0001](./resnet18-xnor-binary-onnx-0001/description/resnet18-xnor-binary-onnx-0001.md)                               | -                    | -          |

## Legal Information
[*] Other names and brands may be claimed as the property of others.
