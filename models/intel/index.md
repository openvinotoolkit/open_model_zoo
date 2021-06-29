# Overview of OpenVINO&trade; Toolkit Intel's Pre-Trained Models

OpenVINO&trade; toolkit provides a set of Intel pre-trained models
that you can use for learning and demo purposes or for developing deep learning
software. Most recent version is available in the [repo on GitHub](https://github.com/openvinotoolkit/open_model_zoo).
The table [Intel's Pre-Trained Models Device Support](./device_support.md) summarizes devices supported by each model.

The models can be downloaded via [Model Downloader](../../tools/downloader/README.md).

> **TIP**: You also can download and profile Intel® pretrained models inside the OpenVINO™ [Deep Learning Workbench](@ref workbench_docs_Workbench_DG_Introduction) (DL Workbench).
> [DL Workbench](@ref workbench_docs_Workbench_DG_Introduction) is a platform built upon OpenVINO™ and provides a web-based graphical environment that enables you to optimize, fine-tune, analyze, visualize, and compare
> performance of deep learning models on various Intel® architecture
> configurations. In the DL Workbench, you can use most of OpenVINO™ toolkit components.
> <br>
> Proceed to an [easy installation from Docker](@ref workbench_docs_Workbench_DG_Install_from_Docker_Hub) to get started.

## Object Detection Models

Several detection models can be used to detect a set of the most popular
objects - for example, faces, people, vehicles. Most of the networks are
SSD-based and provide reasonable accuracy/performance trade-offs. Networks that
detect the same types of objects (for example, `face-detection-adas-0001` and
`face-detection-retail-0004`) provide a choice for higher accuracy/wider
applicability at the cost of slower performance, so you can expect a "bigger"
network to detect objects of the same type better.

| Model Name | Complexity (GFLOPs) | Size (Mp) |
|------------|---------------------|-----------|
| [faster-rcnn-resnet101-coco-sparse-60-0001](./faster-rcnn-resnet101-coco-sparse-60-0001/README.md) | 364.21 | 52.79 |
| [face-detection-adas-0001](./face-detection-adas-0001/README.md)     | 2.835 | 1.053 |
| [face-detection-retail-0004](./face-detection-retail-0004/README.md) | 1.067 | 0.588 |
| [face-detection-retail-0005](./face-detection-retail-0005/README.md) | 0.982 | 1.021 |
| [face-detection-0200](./face-detection-0200/README.md) | 0.785   | 1.828  |
| [face-detection-0202](./face-detection-0202/README.md) | 1.767   | 1.842  |
| [face-detection-0204](./face-detection-0204/README.md) | 2.405   | 1.851  |
| [face-detection-0205](./face-detection-0205/README.md) | 2.853   | 2.392  |
| [face-detection-0206](./face-detection-0206/README.md) | 339.597 | 69.920 |
| [person-detection-retail-0002](./person-detection-retail-0002/README.md) | 12.427 | 3.244 |
| [person-detection-retail-0013](./person-detection-retail-0013/README.md) | 2.300  | 0.723 |
| [person-detection-action-recognition-0005](./person-detection-action-recognition-0005/README.md) | 7.140 | 1.951 |
| [person-detection-action-recognition-0006](./person-detection-action-recognition-0006/README.md) | 8.225 | 2.001 |
| [person-detection-action-recognition-teacher-0002](./person-detection-action-recognition-teacher-0002/README.md) | 7.140 | 1.951 |
| [person-detection-raisinghand-recognition-0001](./person-detection-raisinghand-recognition-0001/README.md) | 7.138 | 1.951 |
| [person-detection-0200](./person-detection-0200/README.md) | 0.786   | 1.817  |
| [person-detection-0201](./person-detection-0201/README.md) | 1.768   | 1.817  |
| [person-detection-0202](./person-detection-0202/README.md) | 3.143   | 1.817  |
| [person-detection-0203](./person-detection-0203/README.md) | 6.519   | 2.394  |
| [person-detection-0106](./person-detection-0106/README.md) | 404.264 | 71.565 |
| [pedestrian-detection-adas-0002](./pedestrian-detection-adas-0002/README.md) | 2.836 | 1.165 |
| [pedestrian-and-vehicle-detector-adas-0001](./pedestrian-and-vehicle-detector-adas-0001/README.md) | 3.974 | 1.650 |
| [vehicle-detection-adas-0002](./vehicle-detection-adas-0002/README.md) | 2.798 | 1.079 |
| [vehicle-detection-0200](./vehicle-detection-0200/README.md) | 0.786 | 1.817 |
| [vehicle-detection-0201](./vehicle-detection-0201/README.md) | 1.768 | 1.817 |
| [vehicle-detection-0202](./vehicle-detection-0202/README.md) | 3.143 | 1.817 |
| [person-vehicle-bike-detection-crossroad-0078](./person-vehicle-bike-detection-crossroad-0078/README.md) | 3.964 | 1.178 |
| [person-vehicle-bike-detection-crossroad-1016](./person-vehicle-bike-detection-crossroad-1016/README.md) | 3.560 | 2.887 |
| [person-vehicle-bike-detection-crossroad-yolov3-1020](./person-vehicle-bike-detection-crossroad-yolov3-1020/README.md) | 65.984 | 61.922 |
| [person-vehicle-bike-detection-2000](./person-vehicle-bike-detection-2000/README.md)    | 0.787 | 1.821  |
| [person-vehicle-bike-detection-2001](./person-vehicle-bike-detection-2001/README.md)    | 1.770 | 1.821  |
| [person-vehicle-bike-detection-2002](./person-vehicle-bike-detection-2002/README.md)    | 3.163 | 1.821  |
| [person-vehicle-bike-detection-2003](./person-vehicle-bike-detection-2003/README.md)    | 6.550 | 2.416  |
| [person-vehicle-bike-detection-2004](./person-vehicle-bike-detection-2004/README.md)    | 1.811 | 2.327  |
| [vehicle-license-plate-detection-barrier-0106](./vehicle-license-plate-detection-barrier-0106/README.md) | 0.349 | 0.634 |
| [product-detection-0001](./product-detection-0001/README.md)                            | 3.598 | 3.212  |
| [person-detection-asl-0001](./person-detection-asl-0001/README.md)                      | 0.986 | 1.338  |
| [yolo-v2-ava-0001](./yolo-v2-ava-0001/README.md)                                        | 29.38 | 48.29  |
| [yolo-v2-ava-sparse-35-0001](./yolo-v2-ava-sparse-35-0001/README.md)                    | 29.38 | 48.29  |
| [yolo-v2-ava-sparse-70-0001](./yolo-v2-ava-sparse-70-0001/README.md)                    | 29.38 | 48.29  |
| [yolo-v2-tiny-ava-0001](./yolo-v2-tiny-ava-0001/README.md)                              | 6.975 | 15.12  |
| [yolo-v2-tiny-ava-sparse-30-0001](./yolo-v2-tiny-ava-sparse-30-0001/README.md)          | 6.975 | 15.12  |
| [yolo-v2-tiny-ava-sparse-60-0001](./yolo-v2-tiny-ava-sparse-60-0001/README.md)          | 6.975 | 15.12  |
| [yolo-v2-tiny-vehicle-detection-0001](./yolo-v2-tiny-vehicle-detection-0001/README.md)  | 5.424 | 11.229 |

## Object Recognition Models

Object recognition models are used for classification, regression, and character
recognition. Use these networks after a respective detector (for example,
Age/Gender recognition after Face Detection).

| Model Name | Complexity (GFLOPs) | Size (Mp) |
|------------|---------------------|-----------|
| [age-gender-recognition-retail-0013](./age-gender-recognition-retail-0013/README.md)         | 0.094 | 2.138 |
| [head-pose-estimation-adas-0001](./head-pose-estimation-adas-0001/README.md)                 | 0.105 | 1.911 |
| [license-plate-recognition-barrier-0001](./license-plate-recognition-barrier-0001/README.md) | 0.328 | 1.218 |
| [vehicle-attributes-recognition-barrier-0039](./vehicle-attributes-recognition-barrier-0039/README.md) | 0.126 | 0.626 |
| [vehicle-attributes-recognition-barrier-0042](./vehicle-attributes-recognition-barrier-0042/README.md) | 0.462 | 11.177 |
| [emotions-recognition-retail-0003](./emotions-recognition-retail-0003/README.md) | 0.126 | 2.483 |
| [landmarks-regression-retail-0009](./landmarks-regression-retail-0009/README.md) | 0.021 | 0.191 |
| [facial-landmarks-35-adas-0002](./facial-landmarks-35-adas-0002/README.md)       | 0.042 | 4.595 |
| [person-attributes-recognition-crossroad-0230](./person-attributes-recognition-crossroad-0230/README.md) | 0.174 | 0.735 |
| [person-attributes-recognition-crossroad-0234](./person-attributes-recognition-crossroad-0234/README.md) | 2.167 | 23.510 |
| [person-attributes-recognition-crossroad-0238](./person-attributes-recognition-crossroad-0238/README.md) | 1.034 | 21.797 |
| [gaze-estimation-adas-0002](./gaze-estimation-adas-0002/README.md) | 0.139 | 1.882 |

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

| Model Name | Complexity (GFLOPs) | Size (Mp) |
|------------|---------------------|-----------|
| [face-reidentification-retail-0095](./face-reidentification-retail-0095/README.md)     | 0.588 | 1.107 |
| [person-reidentification-retail-0288](./person-reidentification-retail-0288/README.md) | 0.174 | 0.183 |
| [person-reidentification-retail-0287](./person-reidentification-retail-0287/README.md) | 0.564 | 0.595 |
| [person-reidentification-retail-0286](./person-reidentification-retail-0286/README.md) | 1.170 | 1.234 |
| [person-reidentification-retail-0277](./person-reidentification-retail-0277/README.md) | 1.993 | 2.103 |

## Semantic Segmentation Models

Semantic segmentation is an extension of object detection problem. Instead of
returning bounding boxes, semantic segmentation models return a "painted"
version of the input image, where the "color" of each pixel represents a certain
class. These networks are much bigger than respective object detection networks,
but they provide a better (pixel-level) localization of objects and they can
detect areas with complex shape (for example, free space on the road).

| Model Name | Complexity (GFLOPs) | Size (Mp) |
|------------|---------------------|-----------|
| [road-segmentation-adas-0001](./road-segmentation-adas-0001/README.md)         | 4.770  | 0.184 |
| [semantic-segmentation-adas-0001](./semantic-segmentation-adas-0001/README.md) | 58.572 | 6.686 |
| [unet-camvid-onnx-0001](./unet-camvid-onnx-0001/README.md)                     | 260.1  | 31.03 |
| [icnet-camvid-ava-0001](./icnet-camvid-ava-0001/README.md)                     | 151.82 | 25.45 |
| [icnet-camvid-ava-sparse-30-0001](./icnet-camvid-ava-sparse-30-0001/README.md) | 151.82 | 25.45 |
| [icnet-camvid-ava-sparse-60-0001](./icnet-camvid-ava-sparse-60-0001/README.md) | 151.82 | 25.45 |

## Instance Segmentation Models

Instance segmentation is an extension of object detection and semantic
segmentation problems. Instead of predicting a bounding box around each object
instance instance segmentation model outputs pixel-wise masks for all instances.

| Model Name | Complexity (GFLOPs) | Size (Mp) |
|------------|---------------------|-----------|
| [instance-segmentation-security-0002](./instance-segmentation-security-0002/README.md) | 423.0842 | 48.3732 |
| [instance-segmentation-security-0091](./instance-segmentation-security-0091/README.md) | 828.6324 | 101.236 |
| [instance-segmentation-security-0228](./instance-segmentation-security-0228/README.md) | 147.2352 | 49.8328 |
| [instance-segmentation-security-1039](./instance-segmentation-security-1039/README.md) | 13.9672  | 10.5674 |
| [instance-segmentation-security-1040](./instance-segmentation-security-1040/README.md) | 29.334   | 13.5673 |

## Human Pose Estimation Models

Human pose estimation task is to predict a pose: body skeleton, which consists
of keypoints and connections between them, for every person in an input image or
video.  Keypoints are body joints, i.e. ears, eyes, nose, shoulders, knees, etc.
There are two major groups of such methods: top-down and bottom-up.  The first
detects persons in a given frame, crops or rescales detections, then runs pose
estimation network for every detection. These methods are very accurate. The
second finds all keypoints in a given frame, then groups them by person
instances, thus faster than previous, because network runs once.

| Model Name | Complexity (GFLOPs) | Size (Mp) |
|------------|---------------------|-----------|
| [human-pose-estimation-0001](./human-pose-estimation-0001/README.md) | 15.435  | 4.099  |
| [human-pose-estimation-0005](./human-pose-estimation-0005/README.md) | 5.9393  | 8.1504 |
| [human-pose-estimation-0006](./human-pose-estimation-0006/README.md) | 8.8720  | 8.1504 |
| [human-pose-estimation-0007](./human-pose-estimation-0007/README.md) | 14.3707 | 8.1504 |

## Image Processing

Deep Learning models find their application in various image processing tasks to
increase the quality of the output.

| Model Name | Complexity (GFLOPs) | Size (Mp) |
|------------|---------------------|-----------|
| [single-image-super-resolution-1032](./single-image-super-resolution-1032/README.md) | 11.654 | 0.030 |
| [single-image-super-resolution-1033](./single-image-super-resolution-1033/README.md) | 30.97 | 16.062 | 0.030 |
| [text-image-super-resolution-0001](./text-image-super-resolution-0001/README.md)     | 1.379  | 0.003 |

## Text Detection

Deep Learning models for text detection in various applications.

| Model Name | Complexity (GFLOPs) | Size (Mp) |
|------------|---------------------|-----------|
| [text-detection-0003](./text-detection-0003/README.md)                       | 51.256 | 6.747 |
| [text-detection-0004](./text-detection-0004/README.md)                       | 23.305 | 4.328 |
| [horizontal-text-detection-0001](./horizontal-text-detection-0001/README.md) | 7.718  | 2.259 |

## Text Recognition

Deep Learning models for text recognition in various applications.

| Model Name | Complexity (GFLOPs) | Size (Mp) |
|------------|---------------------|-----------|
| [text-recognition-0012](./text-recognition-0012/README.md)                                 | 1.485   | 5.568 |
| [text-recognition-0014](./text-recognition-0014/README.md)                                 | 0.5442  | 2.839 |
| [text-recognition-0015](./text-recognition-0015/README.md)                                 |         |       |
| encoder | 12.4 | 398 |
| decoder | 0.03 | 4.33 |
| [handwritten-score-recognition-0003](./handwritten-score-recognition-0003/README.md)       | 0.792   | 5.555 |
| [handwritten-japanese-recognition-0001](./handwritten-japanese-recognition-0001/README.md) | 117.136 | 15.31 |
| [handwritten-simplified-chinese-recognition-0001](./handwritten-simplified-chinese-recognition-0001/README.md) | 134.513 | 17.270 |
| [formula-recognition-medium-scan-0001](./formula-recognition-medium-scan-0001/README.md) |    |    |
|   encoder | 16.56 | 1.86 |
|   decoder | 1.69  | 2.56 |
| [formula-recognition-polynomials-handwritten-0001](./formula-recognition-polynomials-handwritten-0001/README.md) |    |    |
|   encoder | 12.8447 | 0.2017 |
|   decoder | 8.6838  | 2.5449 |

## Text Spotting

Deep Learning models for text spotting (simultaneous detection and recognition).

| Model Name | Complexity (GFLOPs) | Size (Mp) |
|------------|---------------------|-----------|
| [text-spotting-0005](./text-spotting-0005/README.md) |         |        |
|   text-spotting-0005-detector                        | 184.495 | 27.010 |
|   text-spotting-0005-recognizer-encoder              | 2.082   | 1.328  |
|   text-spotting-0005-recognizer-decoder              | 0.002   | 0.273  |

## Action Recognition Models

Action Recognition models predict action that is being performed on a short video clip
(tensor formed by stacking sampled frames from input video). Some models (for example `driver-action-recognition-adas-0002` may use precomputed high-level spatial
or spatio-temporal) features (embeddings) from individual clip fragments and then aggregate them in a temporal model
to predict a vector with classification scores. Models that compute embeddings are called *encoder*, while models
that predict an actual labels are called *decoder*.

| Model Name | Complexity (GFLOPs) | Size (Mp) |
|------------|---------------------|-----------|
| [driver-action-recognition-adas-0002](./driver-action-recognition-adas-0002/README.md) |       |        |
|   driver-action-recognition-adas-0002-encoder                                          | 0.676 | 2.863  |
|   driver-action-recognition-adas-0002-decoder                                          | 0.147 | 4.205  |
| [action-recognition-0001](./action-recognition-0001/README.md)                         |       |        |
|   action-recognition-0001-encoder                                                      | 7.340 | 21.276 |
|   action-recognition-0001-decoder                                                      | 0.147 | 4.405  |
| [asl-recognition-0004](./asl-recognition-0004/README.md)                               | 6.660 | 4.133  |
| [common-sign-language-0002](./common-sign-language-0002/README.md)                     | 4.227 | 4.113  |
| [weld-porosity-detection-0001](./weld-porosity-detection-0001/README.md)               | 3.636 | 11.173 |

## Image Retrieval

Deep Learning models for image retrieval (ranking 'gallery' images according to their similarity to some 'probe' image).

| Model Name | Complexity (GFLOPs) | Size (Mp) |
|------------|---------------------|-----------|
| [image-retrieval-0001](./image-retrieval-0001/README.md) | 0.613 | 2.535 |

## Compressed models

Deep Learning compressed models

| Model Name | Complexity (GFLOPs) | Size (Mp) |
|------------|---------------------|-----------|
| [resnet50-binary-0001](./resnet50-binary-0001/README.md)                     | 1.002 | 7.446 |
| [resnet18-xnor-binary-onnx-0001](./resnet18-xnor-binary-onnx-0001/README.md) | -     | -     |

## Question Answering

| Model Name | Complexity (GFLOPs) | Size (Mp) |
|------------|---------------------|-----------|
| [bert-large-uncased-whole-word-masking-squad-0001](./bert-large-uncased-whole-word-masking-squad-0001/README.md) | 246.93 | 333.96 |
| [bert-large-uncased-whole-word-masking-squad-int8-0001](./bert-large-uncased-whole-word-masking-squad-int8-0001/README.md) | 246.93 | 333.96 |
| [bert-large-uncased-whole-word-masking-squad-emb-0001](./bert-large-uncased-whole-word-masking-squad-emb-0001/README.md) | 246.93 (for [1,384] input size) | 333.96 |
| [bert-small-uncased-whole-word-masking-squad-0001](./bert-small-uncased-whole-word-masking-squad-0001/README.md) | 23.9 | 57.94 |
| [bert-small-uncased-whole-word-masking-squad-0002](./bert-small-uncased-whole-word-masking-squad-0002/README.md) | 23.9 | 41.1 |
| [bert-small-uncased-whole-word-masking-squad-int8-0002](./bert-small-uncased-whole-word-masking-squad-int8-0002/README.md) | 23.9 | 41.1 |
| [bert-small-uncased-whole-word-masking-squad-emb-int8-0001](./bert-small-uncased-whole-word-masking-squad-emb-int8-0001/README.md) | 23.9 (for [1,384] input size) | 41.1 |

## Machine Translation

| Model Name | Complexity (GFLOPs) | Size (Mp) |
|------------|---------------------|-----------|
| [machine-translation-nar-en-ru-0001](./machine-translation-nar-en-ru-0001/README.md) | 23.17 | 69.29 |
| [machine-translation-nar-ru-en-0001](./machine-translation-nar-ru-en-0001/README.md) | 23.17 | 69.29 |
| [machine-translation-nar-en-de-0002](./machine-translation-nar-en-de-0002/README.md) | 23.19 | 77.47 |
| [machine-translation-nar-de-en-0002](./machine-translation-nar-de-en-0002/README.md) | 23.19 | 77.47 |

## Text To Speech

Deep Learning models for speech synthesis (mel spectrogram generation and wave form generation).

| Model Name | Complexity (GFLOPs) | Size (Mp) |
|------------|---------------------|-----------|
| [text-to-speech-en-0001](./text-to-speech-en-0001/README.md) |       |        |
|   text-to-speech-en-0001-duration-prediction                 | 15.84 | 13.569 |
|   text-to-speech-en-0001-regression                          | 7.65  | 4.96   |
|   text-to-speech-en-0001-generation                          | 48.38 | 12.77  |

Deep Learning models for speech synthesis (mel spectrogram generation and wave form generation).

| Model Name | Complexity (GFLOPs) | Size (Mp) |
|------------|---------------------|-----------|
| [text-to-speech-en-multi-0001](./text-to-speech-en-multi-0001/README.md) |       |        |
|   text-to-speech-en-multi-0001-duration-prediction                 | 28.75 | 26.18 |
|   text-to-speech-en-multi-0001-regression                          | 7.81  | 5.12  |
|   text-to-speech-en-multi-0001-generation                          | 48.38 | 12.77 |

Deep Learning models for noise suppression.

| Model Name                                                                          | Complexity (GFLOPs)  | Size (Mp)  |
|-------------------------------------------------------------------------------------|--------------------- |----------- |
| [noise-suppression-poconetlike-0001](./noise-suppression-poconetlike-0001/README.md)| 1.2                  | 7.22       |

## Time Series Forecasting

Deep Learning models for time series forecasting.

| Model Name | Complexity (GFLOPs) | Size (Mp) |
|------------|---------------------|-----------|
| [time-series-forecasting-electricity-0001](./time-series-forecasting-electricity-0001/README.md) | 0.40 | 2.26 |

## See Also

* [Open Model Zoo Demos](../../demos/README.md)
* [Model Downloader](../../tools/downloader/README.md)
* [Overview of OpenVINO&trade; Toolkit Public Pre-Trained Models](../public/index.md)

## Legal Information

[*] Other names and brands may be claimed as the property of others.
