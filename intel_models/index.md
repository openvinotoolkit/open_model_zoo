# Overview of OpenVINO&trade; Toolkit Pre-Trained Models

OpenVINO&trade; toolkit distribution includes a set of pre-trained models
that you can use for learning and demo purposes or for developing a deep learning
software.

## Object Detection Models

Several detection models can be used to detect a set of the most popular
objects - for example, faces, people, vehicles. Most of the networks are
SSD-based and provide reasonable accuracy/performance trade-offs. Networks that
detect the same types of objects (for example, `face-detection-adas-0001` and
`face-detection-retail-0004`) provide a choice for higher accuracy/wider
applicability at the cost of slower performance, so you can expect a "bigger"
network to detect objects of the same type better.

| Model Name                                                                                                                                                                          | Complexity (GFLOPs)  | Size (Mp)  | Face  | Person  | Vehicle  | Bike  | License plate  |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |--------------------- |----------- |------ |-------- |--------- |------ |--------------- |
| [face-detection-adas-0001](./face-detection-adas-0001/description/face-detection-adas-0001.md)                               | 2.835                | 1.053      | X     |         |          |       |                |
| [face-detection-retail-0004](./face-detection-retail-0004/description/face-detection-retail-0004.md)                                                        | 1.067                | 0.588      | X     |         |          |       |                |
| [face-person-detection-retail-0002](./face-person-detection-retail-0002/description/face-person-detection-retail-0002.md)                              | 2.757                | 0.791      | X     | X       |          |       |                |
| [person-detection-retail-0001](./person-detection-retail-0001/description/person-detection-retail-0001.md)                                                  | 12.422               | 3.244      |       | X       |          |       |                |
| [person-detection-retail-0013](./person-detection-retail-0013/description/person-detection-retail-0013.md)                                                      | 2.300                | 0.723      |       | X       |          |       |                |
| [person-detection-action-recognition-0003](./person-detection-action-recognition-0003/description/person-detection-action-recognition-0003.md)                              | 4.542                | 1.951      |       | X       |          |       |                |
| [pedestrian-detection-adas-0002](./pedestrian-detection-adas-0002/description/pedestrian-detection-adas-0002.md)                                   | 2.836                | 1.165      |       | X       |          |       |                |
| [pedestrian-and-vehicle-detector-adas-0001](./pedestrian-and-vehicle-detector-adas-0001/description/pedestrian-and-vehicle-detector-adas-0001.md) | 3.974                | 1.650      |       | X       | X        |       |                |
| [vehicle-detection-adas-0002](./vehicle-detection-adas-0002/description/vehicle-detection-adas-0002.md)                                            | 2.798                | 1.079      |       |         | X        |       |                |
| [person-vehicle-bike-detection-crossroad-0078](./person-vehicle-bike-detection-crossroad-0078/description/person-vehicle-bike-detection-crossroad-0078.md)                               | 3.964                | 1.178      |       | X       | X        | X     |                |
| [vehicle-license-plate-detection-barrier-0106](./vehicle-license-plate-detection-barrier-0106/description/vehicle-license-plate-detection-barrier-0106.md)                                    | 0.349                | 0.634      |       |         | X        |       | X              |

## Object Recognition Models

Object recognition models are used for classification, regression, and character
recognition. Use these networks after a respective detector (for example,
Age/Gender recognition after Face Detection).

| Model Name                                                                                                                                                                                      | Complexity (GFLOPs)  | Size (Mp)  |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------- |----------- |
| [age-gender-recognition-retail-0013](./age-gender-recognition-retail-0013/description/age-gender-recognition-retail-0013.md)                                                                    | 0.094                | 2.138      |
| [head-pose-estimation-adas-0001](./head-pose-estimation-adas-0001/description/head-pose-estimation-adas-0001.md)                                                                | 0.105                | 1.911      |
| [license-plate-recognition-barrier-0001](./license-plate-recognition-barrier-0001/description/license-plate-recognition-barrier-0001.md)                                           | 0.328                | 1.218      |
| [vehicle-attributes-recognition-barrier-0039](./vehicle-attributes-recognition-barrier-0039/description/vehicle-attributes-recognition-barrier-0039.md)                                 | 0.126                | 0.626      |
| [emotions-recognition-retail-0003](./emotions-recognition-retail-0003/description/emotions-recognition-retail-0003.md)                                                         | 0.126                | 2.483      |
| [landmarks-regression-retail-0009](./landmarks-regression-retail-0009/description/landmarks-regression-retail-0009.md)                                                          | 0.021                | 0.191      |
| [facial-landmarks-35-adas-0001](./facial-landmarks-35-adas-0001/description/facial-landmarks-35-adas-0001.md)                                     | 0.042                | 4.595      |
| [person-attributes-recognition-crossroad-0031](./person-attributes-recognition-crossroad-0031/description/person-attributes-recognition-crossroad-0031.md) | 0.219                | 1.102      |

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

| Model Name                                                                                                                                              | Complexity (GFLOPs)  | Size (Mp)  | Pairwise accuracy |
|---------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------- |----------- |------------------ |
| [person-reidentification-retail-0031](./person-reidentification-retail-0031/description/person-reidentification-retail-0031.md)   | 0.028                | 0.280      | 92.11%            |
| [person-reidentification-retail-0079](./person-reidentification-retail-0079/description/person-reidentification-retail-0079.md)   | 0.124                | 0.820      | 92.93%            |
| [person-reidentification-retail-0076](./person-reidentification-retail-0076/description/person-reidentification-retail-0076.md)   | 0.594                | 0.820      | 93.35%            |
| [face-reidentification-retail-0071](./face-reidentification-retail-0071/description/face-reidentification-retail-0071.md)          | 0.586                | 1.059      | 99.33%            |


## Semantic Segmentation Models

Semantic segmentation is an extension of object detection problem. Instead of
returning bounding boxes, semantic segmentation models return a "painted"
version of the input image, where the "color" of each pixel represents a certain
class. These networks are much bigger than respective object detection networks,
but they provide a better (pixel-level) localization of objects and they can
detect areas with complex shape (for example, free space on the road).

| Model Name                                                                                                                                     | Complexity (GFLOPs)  | Size (Mp)  |
|------------------------------------------------------------------------------------------------------------------------------------------------|--------------------- |----------- |
| [road-segmentation-adas-0001](./road-segmentation-adas-0001/description/road-segmentation-adas-0001.md)                          | 4.770                | 0.184      |
| [semantic-segmentation-adas-0001](./semantic-segmentation-adas-0001/description/semantic-segmentation-adas-0001.md) | 58.572               | 6.686      |

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
| [human-pose-estimation-0001](./human-pose-estimation-0001/description/human-pose-estimation-0001.md) | 15.435               | 4.099      |

## Image Processing

Deep Learning models find their application in various image processing tasks to
increase the quality of the output.

| Model Name                                                                                                                                     | Complexity (GFLOPs)  | Size (Mp)  |
|------------------------------------------------------------------------------------------------------------------------------------------------|--------------------- |----------- |
| [single-image-super-resolution-0034](./single-image-super-resolution-0034/description/single-image-super-resolution-0034.md)                     | 39.713               | 0.363      |

## Legal Information
[*] Other names and brands may be claimed as the property of others.
