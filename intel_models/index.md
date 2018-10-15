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
| [face-detection-adas-0001](./face-detection-adas-0001/description/face-detection-adas-0001.md)                               | 2.8                  | 1.1        | X     |         |          |       |                |
| [face-detection-retail-0004](./face-detection-retail-0004/description/face-detection-retail-0004.md)                                                        | 1.1                  | 0.6        | X     |         |          |       |                |
| [face-person-detection-retail-0002](./face-person-detection-retail-0002/description/face-person-detection-retail-0002.md)                              | 2.8                  | 0.8        | X     | X       |          |       |                |
| [person-detection-retail-0001](./person-detection-retail-0001/description/person-detection-retail-0001.md)                                                  | 12.4                 | 3.2        |       | X       |          |       |                |
| [person-detection-retail-0013](./person-detection-retail-0013/description/person-detection-retail-0013.md)                                                      | 2.3                  | 0.7        |       | X       |          |       |                |
| [person-detection-action-recognition-0001](./person-detection-action-recognition-0001/description/person-detection-action-recognition-0001.md)                             | 4.5                  | 1.9        |       | X       |          |       |                |
| [pedestrian-detection-adas-0002](./pedestrian-detection-adas-0002/description/pedestrian-detection-adas-0002.md)                                   | 2.8                  | 1.2        |       | X       |          |       |                |
| [pedestrian-and-vehicle-detector-adas-0001](./pedestrian-and-vehicle-detector-adas-0001/description/pedestrian-and-vehicle-detector-adas-0001.md) | 4.0                  | 1.6        |       | X       | X        |       |                |
| [vehicle-detection-adas-0002](./vehicle-detection-adas-0002/description/vehicle-detection-adas-0002.md)                                            | 2.8                  | 1.1        |       |         | X        |       |                |
| [person-vehicle-bike-detection-crossroad-0078](./person-vehicle-bike-detection-crossroad-0078/description/person-vehicle-bike-detection-crossroad-0078.md)                               | 3.9                  | 1.2        |       | X       | X        | X     |                |
| [vehicle-license-plate-detection-barrier-0106](./vehicle-license-plate-detection-barrier-0106/description/vehicle-license-plate-detection-barrier-0106.md)                                    | 0.4                  | 0.6        |       |         | X        |       | X              |

## Object Recognition Models

Object recognition models are used for classification, regression, and character
recognition. Use these networks after a respective detector (for example,
Age/Gender recognition after Face Detection).

| Model Name                                                                                                                                                                                      | Complexity (GFLOPs)  | Size (Mp)  |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------- |----------- |
| [age-gender-recognition-retail-0013](./age-gender-recognition-retail-0013/description/age-gender-recognition-retail-0013.md)                                                                    | 0.09                 | 2.1        |
| [head-pose-estimation-adas-0001](./head-pose-estimation-adas-0001/description/head-pose-estimation-adas-0001.md)                                                                | 0.1                  | 1.9        |
| [license-plate-recognition-barrier-0001](./license-plate-recognition-barrier-0001/description/license-plate-recognition-barrier-0001.md)                                           | 0.33                 | 1.2        |
| [vehicle-attributes-recognition-barrier-0039](./vehicle-attributes-recognition-barrier-0039/description/vehicle-attributes-recognition-barrier-0039.md)                                 | 0.13                 | 0.6        |
| [emotions-recognition-retail-0003](./emotions-recognition-retail-0003/description/emotions-recognition-retail-0003.md)                                                         | 0.13                 | 2.5        |
| [landmarks-regression-retail-0001](./landmarks-regression-retail-0001/description/landmarks-regression-retail-0001.md)                                                         | 0.02                 | 0.2        |
| [person-attributes-recognition-crossroad-0031](./person-attributes-recognition-crossroad-0031/description/person-attributes-recognition-crossroad-0031.md) | 0.22                 | 1.1        |

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
| [person-reidentification-retail-0031](./person-reidentification-retail-0031/description/person-reidentification-retail-0031.md)   | 0.03                 | 0.28       | 92.11%            |
| [person-reidentification-retail-0079](./person-reidentification-retail-0079/description/person-reidentification-retail-0079.md)   | 0.12                 | 0.82       | 92.93%            |
| [person-reidentification-retail-0076](./person-reidentification-retail-0076/description/person-reidentification-retail-0076.md)   | 0.58                 | 0.82       | 93.35%            |
| [face-reidentification-retail-0001](./face-reidentification-retail-0001/description/face-reidentification-retail-0001.md)                  | 0.19                 | 0.59       | 98.92%            |


## Semantic Segmentation Models

Semantic segmentation is an extension of object detection problem. Instead of
returning bounding boxes, semantic segmentation models return a "painted"
version of the input image, where the "color" of each pixel represents a certain
class. These networks are much bigger than respective object detection networks,
but they provide a better (pixel-level) localization of objects and they can
detect areas with complex shape (for example, free space on the road).

| Model Name                                                                                                                                     | Complexity (GFLOPs)  | Size (Mp)  |
|------------------------------------------------------------------------------------------------------------------------------------------------|--------------------- |----------- |
| [road-segmentation-adas-0001](./road-segmentation-adas-0001/description/road-segmentation-adas-0001.md)                          | 4.7                  | 0.18       |
| [semantic-segmentation-adas-0001](./semantic-segmentation-adas-0001/description/semantic-segmentation-adas-0001.md) | 57.9                 | 6.7        |

## Legal Information
[*] Other names and brands may be claimed as the property of others.
