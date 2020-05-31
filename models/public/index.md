# Overview of OpenVINO&trade; Toolkit Public Models

OpenVINO&trade; toolkit provides a set of public models
that you can use for learning and demo purposes or for developing deep learning
software. Most recent version is available in the [repo on Github](https://github.com/opencv/open_model_zoo).

The models can be downloaded via Model Downloader
(`<OPENVINO_INSTALL_DIR>/deployment_tools/open_model_zoo/tools/downloader`).

## Classification

| Model Name        | Implementation | OMZ Model Name | Accuracy | GFlops | mParams |
| ----------------- | ---------------| -------------- | -------- | ------ | ------- |
| AlexNet           | [Caffe\*](./alexnet/alexnet.md)       | alexnet   | | 1.5 | 60.965 |
| CaffeNet          | [Caffe\*](./caffenet/caffenet.md)     | caffenet  | | 1.5 | 60.965 |
| DenseNet 121      | [Caffe\*](./densenet-121/densenet-121.md)<br>[TensorFlow\*](./densenet-121-tf/densenet-121-tf.md)<br>[Caffe2\*](./densenet-121-caffe2/densenet-121-caffe2.md)  | densenet-121<br>densenet-121-tf<br>densenet-121-caffe2 | | 5.289~5.724    | 7.971 |
| DenseNet 161      | [Caffe\*](./densenet-161/densenet-161.md)<br> [TensorFlow\*](./densenet-161-tf/densenet-161-tf.md) | densenet-161<br>densenet-161-tf | | 14.128~15.561  | 28.666 |
| DenseNet 169      | [Caffe\*](./densenet-169/densenet-169.md)<br>[TensorFlow\*](./densenet-169-tf/densenet-169-tf.md)  | densenet-169<br>densenet-169-tf | | 6.16~6.788 | 14.139 |
| DenseNet 201      | [Caffe\*](./densenet-201/densenet-201.md) | densenet-201 | | 8.673  | 20.001  |
| EfficientNet B0   | [TensorFlow\*](./efficientnet-b0/efficientnet-b0.md)<br>[PyTorch\*](./efficientnet-b0-pytorch/efficientnet-b0-pytorch.md) | efficientnet-b0<br>efficientnet-b0-pytorch | 75.70/92.76<br>76.91/93.21 | 0.819 | 5.268 |
| EfficientNet B0 AutoAugment  | [TensorFlow\*](./efficientnet-b0_auto_aug/efficientnet-b0_auto_aug.md) | efficientnet-b0_auto_aug | 76.43/93.04 | 0.819 | 5.268 |
| EfficientNet B5   | [TensorFlow\*](./efficientnet-b5/efficientnet-b5.md)<br>[PyTorch\*](./efficientnet-b5-pytorch/efficientnet-b5-pytorch.md) | efficientnet-b5<br>efficientnet-b5-pytorch | 83.33/96.67<br>83.69/96.71 | 21.252 | 30.303 |
| EfficientNet B7   | [PyTorch\*](./efficientnet-b7-pytorch/efficientnet-b7-pytorch.md) | efficientnet-b7-pytorch | 84.42/96.91 | 77.618  | 66.193 |
| EfficientNet B7 AutoAugment  | [TensorFlow\*](./efficientnet-b7_auto_aug/efficientnet-b7_auto_aug.md) | efficientnet-b7_auto_aug | 84.68/97.09 | 77.618  | 66.193 |
| HBONet 1.0        | [PyTorch\*](./hbonet-1.0/hbonet-1.0.md)   | hbonet-1.0  | 73.1/91.0 | 0.305 |
| HBONet 0.5        | [PyTorch\*](./hbonet-0.5/hbonet-0.5.md)   | hbonet-0.5  | 67.0/86.9 | 0.096 |
| HBONet 0.25       | [PyTorch\*](./hbonet-0.25/hbonet-0.25.md) | hbonet-0.25 | 57.3/79.8 | 0.037 |
| Inception (GoogleNet) V1 | [Caffe\*](./googlenet-v1/googlenet-v1.md)<br>[TensorFlow\*](./googlenet-v1-tf/googlenet-v1-tf.md) | googlenet-v1<br>googlenet-v1-tf | | 3.016~3.266 | 6.619~6.999 |
| Inception (GoogleNet) V2 | [Caffe\*](./googlenet-v2/googlenet-v2.md)<br>[TensorFlow\*](./googlenet-v2-tf/googlenet-v2-tf.md) | googlenet-v2<br>googlenet-v2-tf | | 4.058 | 11.185 |
| Inception (GoogleNet) V3 | [TensorFlow\*](./googlenet-v3/googlenet-v3.md)<br>[PyTorch\*](./googlenet-v3-pytorch/googlenet-v3-pytorch.md) | googlenet-v3 <br> googlenet-v3-pytorch |  | 11.469 | 23.817 |
| Inception (GoogleNet) V4 | [TensorFlow\*](./googlenet-v4-tf/googlenet-v4-tf.md) | googlenet-v4-tf | | 24.584 | 42.648 |
| Inception-ResNet V2      | [TensorFlow\*](./inception-resnet-v2-tf/inception-resnet-v2-tf.md) | inception-resnet-v2-tf | | 22.227 | 30.223 |
| MobileNet V1 0.25 128 | [Caffe\*](./mobilenet-v1-0.25-128/mobilenet-v1-0.25-128.md) | mobilenet-v1-0.25-128 | | 0.028 | 0.468 |
| MobileNet V1 0.5 160 | [Caffe\*](./mobilenet-v1-0.50-160/mobilenet-v1-0.50-160.md)  | mobilenet-v1-0.50-160 | | 0.156 | 1.327 |
| MobileNet V1 0.5 224 | [Caffe\*](./mobilenet-v1-0.50-224/mobilenet-v1-0.50-224.md)  | mobilenet-v1-0.50-224 | | 0.304 | 1.327 |
| MobileNet V1 1.0 224 | [Caffe\*](./mobilenet-v1-1.0-224/mobilenet-v1-1.0-224.md)<br>[TensorFlow\*](./mobilenet-v1-1.0-224-tf/mobilenet-v1-1.0-224-tf.md) | mobilenet-v1-1.0-224 <br> mobilenet-v1-1.0-224-tf | | 1.148 | 4.221 |
| MobileNet V2 1.0 224 | [Caffe\*](./mobilenet-v2/mobilenet-v2.md)<br>[TensorFlow\*](./mobilenet-v2-1.0-224/mobilenet-v2-1.0-224.md)<br>[PyTorch\*](./mobilenet-v2-pytorch/mobilenet-v2-pytorch.md) | mobilenet-v2 <br>  mobilenet-v2-1.0-224 <br> mobilenet-v2-pytorch | | 0.615~0.876 | 3.489 |
| MobileNet V2 1.4 224 | [TensorFlow\*](./mobilenet-v2-1.4-224/mobilenet-v2-1.4-224.md)  | mobilenet-v2-1.4-224 | | 1.183  | 6.087 |
| MobileNet V3 Small 1.0 | [TensorFlow\*](./mobilenet-v3-small-1.0-224-tf/mobilenet-v3-small-1.0-224-tf.md) | mobilenet-v3-small-1.0-224-tf | 67.36/87.45 | 0.121 | 2.537 |
| MobileNet V3 Large 1.0 | [TensorFlow\*](./mobilenet-v3-large-1.0-224-tf/mobilenet-v3-large-1.0-224-tf.md) | mobilenet-v3-large-1.0-224-tf | 75.70/92.76 | 0.4536 | 5.4721 |
| ResNet 34            | [PyTorch\*](./resnet-34-pytorch/resnet-34-pytorch.md) | resnet-34-pytorch | 73.30/91.42 | 7.3409 | 21.7892 |
| ResNet 50            | [Caffe\*](./resnet-50/resnet-50.md) <br> [PyTorch\*](./resnet-50-pytorch/resnet-50-pytorch.md)<br>[Caffe2\*](./resnet-50-caffe2/resnet-50-caffe2.md)<br>[TensorFlow\*](./resnet-50-tf/resnet-50-tf.md) | resnet-50 <br> resnet-50-pytorch<br>resnet-50-caffe2<br>resnet-50-tf | | 6.996~8.216 | 25.53 |
| ResNet 101           | [Caffe\*](./resnet-101/resnet-101.md) | resnet-101 | | 14.441 | 44.496 |
| ResNet 152           | [Caffe\*](./resnet-152/resnet-152.md) | resnet-152 | | 21.89 | 60.117 |
| SE-Inception         | [Caffe\*](./se-inception/se-inception.md)      | se-inception | | 4.091 | 11.922 |
| SE-ResNet 50         | [Caffe\*](./se-resnet-50/se-resnet-50.md)      | se-resnet-50 | | 7.775 | 28.061 |
| SE-ResNet 101        | [Caffe\*](./se-resnet-101/se-resnet-101.md)    | se-resnet-101  | | 15.239 | 49.274  |
| SE-ResNet 152        | [Caffe\*](./se-resnet-152/se-resnet-152.md)    | se-resnet-152 | | 22.709 | 66.746 |
| SE-ResNeXt 50        | [Caffe\*](./se-resnext-50/se-resnext-50.md)    | se-resnext-50 | | 8.533 | 27.526|
| SE-ResNeXt 101       | [Caffe\*](./se-resnext-101/se-resnext-101.md)  | se-resnext-101 | | 16.054 | 48.886 |
| SqueezeNet v1.0      | [Caffe\*](./squeezenet1.0/squeezenet1.0.md)    | squeezenet1.0| | 1.737 | 1.248 |
| SqueezeNet v1.1      | [Caffe\*](./squeezenet1.1/squeezenet1.1.md)<br>[Caffe2\*](./squeezenet1.1-caffe2/squeezenet1.1-caffe2.md)    | squeezenet1.1<br>squeezenet1.1-caffe2| | 0.785 | 1.236 |
| VGG 16               | [Caffe\*](./vgg16/vgg16.md) | vgg16 | | 30.974 | 138.358 |
| VGG 19               | [Caffe\*](./vgg19/vgg19.md)<br>[Caffe2\*](./vgg19-caffe2/vgg19-caffe2.md) | vgg19<br>vgg19-caffe2 | | 39.3 | 143.667  |

**Octave Convolutions Networks**

This is are modifications of networks using Octave Convolutions. More details can be found [here](https://arxiv.org/abs/1904.05049).

| Model Name                | Implementation | OMZ Model Name | Accuracy | GFlops | mParams |
| ------------------------- | ---------------| -------------- | -------- | ------ | ------- |
| DenseNet 121, alpha=0.125 | [MXNet\*](./octave-densenet-121-0.125/octave-densenet-121-0.125.md)   | octave-densenet-121-0.125 | | 4.883 | 7.977 |
| ResNet 26, alpha=0.25     | [MXNet\*](./octave-resnet-26-0.25/octave-resnet-26-0.25.md)           | octave-resnet-26-0.25     | | 3.768 | 15.99 |
| ResNet 50, alpha=0.125    | [MXNet\*](./octave-resnet-50-0.125/octave-resnet-50-0.125.md)         | octave-resnet-50-0.125    | | 7.221 | 25.551 |
| ResNet 101, alpha=0.125   | [MXNet\*](./octave-resnet-101-0.125/octave-resnet-101-0.125.md)       | octave-resnet-101-0.125   | | 13.387 | 44.543 |
| ResNet 200, alpha=0.125   | [MXNet\*](./octave-resnet-200-0.125/octave-resnet-200-0.125.md)       | octave-resnet-200-0.125   | | 25.407 | 64.667 |
| ResNeXt 50, alpha=0.25    | [MXNet\*](./octave-resnext-50-0.25/octave-resnext-50-0.25.md)         | octave-resnext-50-0.25    | | 6.444 | 25.02 |
| ResNeXt 101, alpha=0.25   | [MXNet\*](./octave-resnext-101-0.25/octave-resnext-101-0.25.md)       | octave-resnext-101-0.25   | | 11.521 | 44.169 |
| SE-ResNet 50, alpha=0.125 | [MXNet\*](./octave-se-resnet-50-0.125/octave-se-resnet-50-0.125.md)   | octave-se-resnet-50-0.125 | | 7.246 | 28.082 |

## Segmentation

Semantic segmentation is an extension of object detection problem. Instead of
returning bounding boxes, semantic segmentation models return a "painted"
version of the input image, where the "color" of each pixel represents a certain
class. These networks are much bigger than respective object detection networks,
but they provide a better (pixel-level) localization of objects and they can
detect areas with complex shape.

### Semantic Segmentation

| Model Name                | Implementation | OMZ Model Name | GFlops | mParams |
| ------------------------- | -------------- | -------------- | ------ | ------- |
| DeepLab V3                | [TensorFlow\*](./deeplabv3/deeplabv3.md) | deeplabv3 | 11.469 | 23.819 |

### Instance Segmentation

Instance segmentation is an extension of object detection and semantic
segmentation problems. Instead of predicting a bounding box around each object
instance instance segmentation model outputs pixel-wise masks for all instances.

| Model Name                | Implementation | OMZ Model Name | GFlops | mParams |
| ------------------------- | -------------- | -------------- | ------ | ------- |
| Mask R-CNN Inception ResNet V2 | [TensorFlow\*](./mask_rcnn_inception_resnet_v2_atrous_coco/mask_rcnn_inception_resnet_v2_atrous_coco.md) | mask_rcnn_inception_resnet_v2_atrous_coco | 675.314 | 92.368 |
| Mask R-CNN Inception V2   | [TensorFlow\*](./mask_rcnn_inception_v2_coco/mask_rcnn_inception_v2_coco.md) | mask_rcnn_inception_v2_coco | 54.926 | 21.772 |
| Mask R-CNN ResNet 50      | [TensorFlow\*](./mask_rcnn_resnet50_atrous_coco/mask_rcnn_resnet50_atrous_coco.md) | mask_rcnn_resnet50_atrous_coco| 294.738 | 50.222 |
| Mask R-CNN ResNet 101     | [TensorFlow\*](./mask_rcnn_resnet101_atrous_coco/mask_rcnn_resnet101_atrous_coco.md) | mask_rcnn_resnet101_atrous_coco | 674.58 | 69.188 |

### 3D Semantic Segmentation

| Model Name                | Implementation | OMZ Model Name | GFlops | mParams |
| ------------------------- | -------------- | -------------- | ------ | ------- |
| Brain Tumor Segmentation  | [MXNet\*](./brain-tumor-segmentation-0001/brain-tumor-segmentation-0001.md) | brain-tumor-segmentation-0001 | 409.996 | 38.192 |
| Brain Tumor Segmentation 2  | [PyTorch\*](./brain-tumor-segmentation-0002/brain-tumor-segmentation-0002.md) | brain-tumor-segmentation-0002 | 300.801 | 4.51  |

## Object Detection

Several detection models can be used to detect a set of the most popular
objects - for example, faces, people, vehicles. Most of the networks are
SSD-based and provide reasonable accuracy/performance trade-offs.

| Model Name                | Implementation | OMZ Model Name | GFlops | mParams |
| ------------------------- | -------------- | -------------- | ------ | ------- |
| CTPN                      | [TensorFlow\*](./ctpn/ctpn.md) | ctpn | 55.813 | 17.237 | |
| CenterNet (CTDET with DLAV0) 384x384 | [ONNX\*](./ctdet_coco_dlav0_384/ctdet_coco_dlav0_384.md) | ctdet_coco_dlav0_384| 34.994 | 17.911 | |
| CenterNet (CTDET with DLAV0) 512x512 | [ONNX\*](./ctdet_coco_dlav0_512/ctdet_coco_dlav0_512.md) | ctdet_coco_dlav0_512| 62.211 | 17.911 | |
| FaceBoxes | [PyTorch\*](./faceboxes-pytorch/faceboxes-pytorch.md) | faceboxes-pytorch| 1.8975 | 1.0059 |
| Faster R-CNN with Inception-ResNet v2 | [TensorFlow\*](./faster_rcnn_inception_resnet_v2_atrous_coco/faster_rcnn_inception_resnet_v2_atrous_coco.md) | faster_rcnn_inception_resnet_v2_atrous_coco| 30.687 | 13.307 |
| Faster R-CNN with Inception v2 | [TensorFlow\*](./faster_rcnn_inception_v2_coco/faster_rcnn_inception_v2_coco.md) | faster_rcnn_inception_v2_coco | 30.687 | 13.307 |
| Faster R-CNN with ResNet 50    | [TensorFlow\*](./faster_rcnn_resnet50_coco/faster_rcnn_resnet50_coco.md) | faster_rcnn_resnet50_coco | 57.203 | 29.162 |
| Faster R-CNN with ResNet 101   | [TensorFlow\*](./faster_rcnn_resnet101_coco/faster_rcnn_resnet101_coco.md) |faster_rcnn_resnet101_coco | 112.052 | 48.128 |
| MobileFace Detection V1  | [MXNet\*](./mobilefacedet-v1-mxnet/mobilefacedet-v1-mxnet.md) | mobilefacedet-v1-mxnet| 3.5456 | 7.6828 |
| MTCNN                     | [Caffe\*](./mtcnn/mtcnn.md) | mtcnn-p <br>mtcnn-r <br>mtcnn-o| | |
| Pelee                     | [Caffe\*](./pelee-coco/pelee-coco.md) | pelee-coco | 1.290 | 5.98 |
| RetinaNet with Resnet 50  | [TensorFlow\*](./retinanet-tf/retinanet-tf.md) | retinanet-tf | 238.9469 | 64.9706 |
| R-FCN with Resnet-101     | [TensorFlow\*](./rfcn-resnet101-coco-tf/rfcn-resnet101-coco-tf.md) | rfcn-resnet101-coco-tf | 53.462 | 171.85 |
| SSD 300                   | [Caffe\*](./ssd300/ssd300.md) | ssd300 | 62.815 | 26.285 |
| SSD 512                   | [Caffe\*](./ssd512/ssd512.md) | ssd512 | 180.611 | 27.189 |
| SSD with MobileNet        | [Caffe\*](./mobilenet-ssd/mobilenet-ssd.md) <br>[TensorFlow\*](./ssd_mobilenet_v1_coco/ssd_mobilenet_v1_coco.md) | mobilenet-ssd <br>ssd_mobilenet_v1_coco | 2.316~2.494 | 5.783~6.807 |
| SSD with MobileNet FPN    | [TensorFlow\*](./ssd_mobilenet_v1_fpn_coco/ssd_mobilenet_v1_fpn_coco.md) | ssd_mobilenet_v1_fpn_coco | 123.309 | 36.188 |
| SSD with MobileNet V2     | [TensorFlow\*](./ssd_mobilenet_v2_coco/ssd_mobilenet_v2_coco.md) | ssd_mobilenet_v2_coco | 3.775 | 16.818 |
| SSD lite with MobileNet V2 | [TensorFlow\*](./ssdlite_mobilenet_v2/ssdlite_mobilenet_v2.md)  | ssdlite_mobilenet_v2 | 1.525 | 4.475 |
| SSD with ResNet-50 V1 FPN | [TensorFlow\*](./ssd_resnet50_v1_fpn_coco/ssd_resnet50_v1_fpn_coco.md) | ssd_resnet50_v1_fpn_coco | 178.6807 | 59.9326 |
| SSD with ResNet 34 1200x1200 | [PyTorch\*](./ssd-resnet34-1200-onnx/ssd-resnet34-1200-onnx.md)  | ssd-resnet34-1200-onnx | 433.411 | 20.058  |
| RetinaFace-R50            | [MXNet\*](./retinaface-resnet50/retinaface-resnet50.md) | retinaface-resnet50 | 100.8478 | 29.427 |
| RetinaFace-Anti-Cov     | [MXNet\*](./retinaface-anti-cov/retinaface-anti-cov.md)  | retinaface-anti-cov | 2.7781 | 0.5955 |
| YOLO v1 Tiny              | [TensorFlow.js\*](./yolo-v1-tiny-tf/yolo-v1-tiny-tf.md) | yolo-v1-tiny-tf | 6.9883	 |	15.8587 |
| YOLO v2 Tiny              | [TensorFlow.js\*](./yolo-v2-tiny-tf/yolo-v2-tiny-tf.md) | yolo-v2-tiny-tf | 5.4236	 |	11.2295 |
| YOLO v2                   | [Keras\*](./yolo-v2-tf/yolo-v2-tf.md) | yolo-v2-tf | 63.0301	 |	50.9526 |
| YOLO v3                   | [Keras\*](./yolo-v3-tf/yolo-v3-tf.md) | yolo-v3-tf | 65.9843	 |	61.9221 |

## Object attributes

### Facial Landmarks

| Model Name                | Implementation | OMZ Model Name | GFlops | mParams |
| ------------------------- | -------------- | -------------- | ------ | ------- |
| RetinaFace-R50            | [MXNet\*](./retinaface-resnet50/retinaface-resnet50.md) | retinaface-resnet50 | 100.8478 | 29.427 |
| RetinaFace-Anti-Cov     | [MXNet\*](./retinaface-anti-cov/retinaface-anti-cov.md)  | retinaface-anti-cov | 2.7781 | 0.5955 |

## Face Recognition

| Model Name                | Implementation | OMZ Model Name | GFlops | mParams |
| ------------------------- | -------------- | -------------- |------ | ------- |
| FaceNet                   | [TensorFlow\*](./facenet-20180408-102900/facenet-20180408-102900.md) | facenet-20180408-102900 | 2.846 | 23.469 |
| LResNet34E-IR,ArcFace@ms1m-refine-v1 | [MXNet\*](./face-recognition-resnet34-arcface/face-recognition-resnet34-arcface.md) | face-recognition-resnet34-arcface | 8.934 | 34.129 |
| LResNet50E-IR,ArcFace@ms1m-refine-v1 | [MXNet\*](./face-recognition-resnet50-arcface/face-recognition-resnet50-arcface.md) | face-recognition-resnet50-arcface | 12.637 | 43.576 |
| LResNet100E-IR,ArcFace@ms1m-refine-v2 | [MXNet\*](./face-recognition-resnet100-arcface/face-recognition-resnet100-arcface.md) | face-recognition-resnet100-arcface | 24.209 | 65.131 |
| MobileFaceNet,ArcFace@ms1m-refine-v1 | [MXNet\*](./face-recognition-mobilefacenet-arcface/face-recognition-mobilefacenet-arcface.md) | face-recognition-mobilefacenet-arcface | 0.449 | 0.993 |
| SphereFace                | [Caffe\*](./Sphereface/Sphereface.md) | Sphereface | 3.504 | 22.671 |

## Human Pose Estimation

Human pose estimation task is to predict a pose: body skeleton, which consists
of keypoints and connections between them, for every person in an input image or
video. Keypoints are body joints, i.e. ears, eyes, nose, shoulders, knees, etc.
There are two major groups of such metods: top-down and bottom-up.  The first
detects persons in a given frame, crops or rescales detections, then runs pose
estimation network for every detection. These methods are very accurate. The
second finds all keypoints in a given frame, then groups them by person
instances, thus faster than previous, because network runs once.

| Model Name                    | Implementation                                                                            | OMZ Model Name                | GFlops | mParams |
|------------------------------ | ----------------------------------------------------------------------------------------- | ----------------------------- | ------ | ------- |
| human-pose-estimation-3d-0001 | [PyTorch\*](./human-pose-estimation-3d-0001/description/human-pose-estimation-3d-0001.md) | human-pose-estimation-3d-0001 | 18.998 |  5.074  |
|single-human-pose-estimation-0001| [PyTorch\*](./single-human-pose-estimation-0001/description/single-human-pose-estimation-0001.md) | single-human-pose-estimation-0001 | 60.125 | 33.165 |

## Monocular Depth Estimation

The task of monocular depth estimation is to predict a depth (or inverse depth) map based on a single input image.
Since this task contains - in the general setting - some ambiguity, the resulting depth maps are often only defined up to an unknown scaling factor.

| Model Name                  | Implementation                      | OMZ Model Name                | GFlops    | mParams |
| --------------------------- | ----------------------------------- | ----------------------------- | --------- | ------- |
| midasnet                    | [PyTorch\*](./midasnet/midasnet.md) | midasnet                      | 207.4915  |         |

## Image Inpainting

Image inpainting task is to estimate suitable pixel information to fill holes in images.

| Model Name                | Implementation | OMZ Model Name | Accuracy | GFlops | mParams |
| ------------------------- | ---------------| -------------- | -------- | ------ | ------- |
| GMCNN Inpainting          | [TensorFlow\*](./gmcnn-places2-tf/gmcnn-places2-tf.md) | gmcnn-places2-tf | 33.47Db | | |

## Style Transfer

Style transfer task is to transfer the style of one image to another.

| Model Name                     | Implementation | OMZ Model Name | Accuracy | GFlops | mParams |
| ------------------------------ | ---------------| -------------- | -------- | ------ | ------- |
| fast-neural-style-mosaic-onnx  | [ONNX\*](./fast-neural-style-mosaic-onnx/fast-neural-style-mosaic-onnx.md) | fast-neural-style-mosaic-onnx | 12.04dB | 15.518 | 1.679 |

## Action Recognition

The task of action recognition is to predict action that is being performed on a short video clip
(tensor formed by stacking sampled frames from input video).

| Model Name                | Implementation | OMZ Model Name | GFlops | mParams |
| ------------------------- | ---------------| -------------- | ------ | ------- |
| RGB-I3D, pretrained on ImageNet\* | [TensorFlow\*](./i3d-rgb-tf/i3d-rgb-tf.md) | i3d-rgb-tf | | |

## Legal Information

[*] Other names and brands may be claimed as the property of others.
