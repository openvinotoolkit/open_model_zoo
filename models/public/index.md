# Overview of OpenVINO&trade; Toolkit Public Models

OpenVINO&trade; toolkit provides a set of public models
that you can use for learning and demo purposes or for developing deep learning
software. Most recent version is available in the [repo on Github](https://github.com/opencv/open_model_zoo).

The models can be downloaded via Model Downloader
(`<OPENVINO_INSTALL_DIR>/deployment_tools/open_model_zoo/tools/downloader`).

## Classification

| Model Name                  | Implementation                     | OMZ Model Name | Accuracy | GFlops | mParams |
| --------------------------- | -----------------------------------| -------------- | -------- | ------ | ------- |
| AlexNet                     | Caffe\*                            | [alexnet](./alexnet/alexnet.md)   | 56.598%/79.812% | 1.5 | 60.965 |
| CaffeNet                    | Caffe\*                            | [caffenet](./caffenet/caffenet.md)  | 56.714%/79.916% | 1.5 | 60.965 |
| DenseNet 121                | Caffe\*<br>TensorFlow\*<br>Caffe2\*| [densenet-121](./densenet-121/densenet-121.md)<br>[densenet-121-tf](./densenet-121-tf/densenet-121-tf.md)<br>[densenet-121-caffe2](./densenet-121-caffe2/densenet-121-caffe2.md) | 74.42%/92.136%<br>74.29%/91.98% <br>74.904%/92.192% | 5.289~5.724    | 7.971 |
| DenseNet 161                | Caffe\*<br>TensorFlow\*            | [densenet-161](./densenet-161/densenet-161.md)<br>[densenet-161-tf](./densenet-161-tf/densenet-161-tf.md) | 77.55%/93.92%<br>76.446%/93.228%| 14.128~15.561  | 28.666 |
| DenseNet 169                | Caffe\*<br>TensorFlow\*            | [densenet-169](./densenet-169/densenet-169.md)<br>[densenet-169-tf](./densenet-169-tf/densenet-169-tf.md) | 76.106%/93.106%<br>75.76%/92.81%| 6.16~6.788 | 14.139 |
| DenseNet 201                | Caffe\*                            | [densenet-201](./densenet-201/densenet-201.md)| 76.886%/93.556% | 8.673  | 20.001  |
| EfficientNet B0             | TensorFlow\*<br>PyTorch\*          | [efficientnet-b0](./efficientnet-b0/efficientnet-b0.md)<br>[efficientnet-b0-pytorch](./efficientnet-b0-pytorch/efficientnet-b0-pytorch.md) | 75.70%/92.76%<br>76.91%/93.21% | 0.819 | 5.268 |
| EfficientNet B0 AutoAugment | TensorFlow\*                       | [efficientnet-b0_auto_aug](./efficientnet-b0_auto_aug/efficientnet-b0_auto_aug.md) | 76.43%/93.04% | 0.819 | 5.268 |
| EfficientNet B5             | TensorFlow\*<br>PyTorch\*          | [efficientnet-b5](./efficientnet-b5/efficientnet-b5.md)<br>[efficientnet-b5-pytorch](./efficientnet-b5-pytorch/efficientnet-b5-pytorch.md) | 83.33%/96.67%<br>83.69%/96.71% | 21.252 | 30.303 |
| EfficientNet B7             | PyTorch\*                          | [efficientnet-b7-pytorch](./efficientnet-b7-pytorch/efficientnet-b7-pytorch.md) | 84.42%/96.91% | 77.618  | 66.193 |
| EfficientNet B7 AutoAugment | TensorFlow\*                       | [efficientnet-b7_auto_aug](./efficientnet-b7_auto_aug/efficientnet-b7_auto_aug.md) | 84.68%/97.09% | 77.618  | 66.193 |
| HBONet 1.0                  | PyTorch\*                          | [hbonet-1.0](./hbonet-1.0/hbonet-1.0.md)  | 73.1%/91.0% | 0.6208 | 4.5443 |
| HBONet 0.5                  | PyTorch\*                          | [hbonet-0.5](./hbonet-0.5/hbonet-0.5.md)  | 67.0%/86.9% | 0.1977 | 2.5287 |
| HBONet 0.25                 | PyTorch\*                          | [hbonet-0.25](./hbonet-0.25/hbonet-0.25.md) | 57.3%/79.8% | 0.0758 | 1.9299 |
| Inception (GoogleNet) V1    | Caffe\*<br>TensorFlow\*            | [googlenet-v1](./googlenet-v1/googlenet-v1.md)<br>[googlenet-v1-tf](./googlenet-v1-tf/googlenet-v1-tf.md) | 68.928%/89.144%<br>69.814%/89.6% | 3.016~3.266 | 6.619~6.999 |
| Inception (GoogleNet) V2    | Caffe\*<br>TensorFlow\*            | [googlenet-v2](./googlenet-v2/googlenet-v2.md)<br>[googlenet-v2-tf](./googlenet-v2-tf/googlenet-v2-tf.md) | 72.024%/90.844%<br>74.084%/91.798%| 4.058 | 11.185 |
| Inception (GoogleNet) V3    | TensorFlow\*<br>PyTorch\*          | [googlenet-v3](./googlenet-v3/googlenet-v3.md) <br> [googlenet-v3-pytorch](./googlenet-v3-pytorch/googlenet-v3-pytorch.md) | 77.904%/93.808%<br>77.696%/93.696% | 11.469 | 23.817 |
| Inception (GoogleNet) V4    | TensorFlow\*                       | [googlenet-v4-tf](./googlenet-v4-tf/googlenet-v4-tf.md) | 80.204%/95.21% | 24.584 | 42.648 |
| Inception-ResNet V2         | TensorFlow\*                       | [inception-resnet-v2-tf](./inception-resnet-v2-tf/inception-resnet-v2-tf.md) | 80.14%/95.10% | 22.227 | 30.223 |
| MobileNet V1 0.25 128       | Caffe\*                            | [mobilenet-v1-0.25-128](./mobilenet-v1-0.25-128/mobilenet-v1-0.25-128.md)  | 40.54%/65% | 0.028 | 0.468 |
| MobileNet V1 0.5 160        | Caffe\*                            | [mobilenet-v1-0.50-160](./mobilenet-v1-0.50-160/mobilenet-v1-0.50-160.md) | 59.86%/82.04% | 0.156 | 1.327 |
| MobileNet V1 0.5 224        | Caffe\*                            | [mobilenet-v1-0.50-224](./mobilenet-v1-0.50-224/mobilenet-v1-0.50-224.md) | 63.042%/84.934%| 0.304 | 1.327 |
| MobileNet V1 1.0 224        | Caffe\*<br>TensorFlow\*            | [mobilenet-v1-1.0-224](./mobilenet-v1-1.0-224/mobilenet-v1-1.0-224.md)<br>[mobilenet-v1-1.0-224-tf](./mobilenet-v1-1.0-224-tf/mobilenet-v1-1.0-224-tf.md)| 69.496%/89.224%<br>71.03%/89.94% | 1.148 | 4.221 |
| MobileNet V2 1.0 224        | Caffe\*<br>TensorFlow\*<br>PyTorch\*| [mobilenet-v2](./mobilenet-v2/mobilenet-v2.md) <br>[mobilenet-v2-1.0-224](./mobilenet-v2-1.0-224/mobilenet-v2-1.0-224.md)<br>[mobilenet-v2-pytorch](./mobilenet-v2-pytorch/mobilenet-v2-pytorch.md) | 71.218%/90.178%<br>71.85%/90.69%<br>71.81%/90.396% | 0.615~0.876 | 3.489 |
| MobileNet V2 1.4 224        | TensorFlow\*                       | [mobilenet-v2-1.4-224](./mobilenet-v2-1.4-224/mobilenet-v2-1.4-224.md) | 74.09%/91.97% | 1.183  | 6.087 |
| MobileNet V3 Small 1.0      | TensorFlow\*                       | [mobilenet-v3-small-1.0-224-tf](./mobilenet-v3-small-1.0-224-tf/mobilenet-v3-small-1.0-224-tf.md) | 67.36%/87.45% | 0.121 | 2.537 |
| MobileNet V3 Large 1.0      | TensorFlow\*                       | [mobilenet-v3-large-1.0-224-tf](./mobilenet-v3-large-1.0-224-tf/mobilenet-v3-large-1.0-224-tf.md) | 75.70%/92.76% | 0.4536 | 5.4721 |
| DenseNet 121, alpha=0.125   | MXNet\*                            | [octave-densenet-121-0.125](./octave-densenet-121-0.125/octave-densenet-121-0.125.md) | 76.066%/93.044% | 4.883 | 7.977 |
| ResNet 26, alpha=0.25       | MXNet\*                            | [octave-resnet-26-0.25](./octave-resnet-26-0.25/octave-resnet-26-0.25.md)     | 76.076%/92.584%| 3.768 | 15.99 |
| ResNet 50, alpha=0.125      | MXNet\*                            | [octave-resnet-50-0.125](./octave-resnet-50-0.125/octave-resnet-50-0.125.md)    | 78.19%/93.862% | 7.221 | 25.551 |
| ResNet 101, alpha=0.125     | MXNet\*                            | [octave-resnet-101-0.125](./octave-resnet-101-0.125/octave-resnet-101-0.125.md)   | 79.182%/94.42% | 13.387 | 44.543 |
| ResNet 200, alpha=0.125     | MXNet\*                            | [octave-resnet-200-0.125](./octave-resnet-200-0.125/octave-resnet-200-0.125.md)   | 79.99%/94.866%| 25.407 | 64.667 |
| ResNeXt 50, alpha=0.25      | MXNet\*                            | [octave-resnext-50-0.25](./octave-resnext-50-0.25/octave-resnext-50-0.25.md)    | 78.772%/94.18% | 6.444 | 25.02 |
| ResNeXt 101, alpha=0.25     | MXNet\*                            | [octave-resnext-101-0.25](./octave-resnext-101-0.25/octave-resnext-101-0.25.md)   | 79.556%/94.444% | 11.521 | 44.169 |
| SE-ResNet 50, alpha=0.125   | MXNet\*                            | [octave-se-resnet-50-0.125](./octave-se-resnet-50-0.125/octave-se-resnet-50-0.125.md) | 78.706%/94.09% | 7.246 | 28.082 |
| open-closed-eye-0001        | PyTorch\*                          | [open-closed-eye-0001](./open-closed-eye-0001/description/open-closed-eye-0001.md) |  95.84%   | 0.0014 | 0.0113|
| ResNet 18                   | PyTorch\*                          | [resnet-18-pytorch](./resnet-18-pytorch/resnet-18-pytorch.md) | 69.754%/89.088% | 3.637 |  11.68 |
| ResNet 34                   | PyTorch\*                          | [resnet-34-pytorch](./resnet-34-pytorch/resnet-34-pytorch.md) | 73.30%/91.42% | 7.3409 | 21.7892 |
| ResNet 50                   | Caffe\*<br>PyTorch\*<br>Caffe2\*<br>TensorFlow\*| [resnet-50](./resnet-50/resnet-50.md) <br> [resnet-50-pytorch](./resnet-50-pytorch/resnet-50-pytorch.md)<br>[resnet-50-caffe2](./resnet-50-caffe2/resnet-50-caffe2.md)<br>[resnet-50-tf](./resnet-50-tf/resnet-50-tf.md)| 75.168%/92.212%<br>76.128%/92.858%<br>76.38%/93.188%<br>76.17%/92.98% | 6.996~8.216 | 25.53 |
| ResNet 101                  | Caffe\*                            | [resnet-101](./resnet-101/resnet-101.md) | 76.364%/92.902% | 14.441 | 44.496 |
| ResNet 152                  | Caffe\*                            | [resnet-152](./resnet-152/resnet-152.md) | 76.802%/93.192% | 21.89 | 60.117 |
| SE-Inception                | Caffe\*                            | [se-inception](./se-inception/se-inception.md) | 75.996%/92.964% | 4.091 | 11.922 |
| SE-ResNet 50                | Caffe\*                            | [se-resnet-50](./se-resnet-50/se-resnet-50.md) | 77.596%/93.85% | 7.775 | 28.061 |
| SE-ResNet 101               | Caffe\*                            | [se-resnet-101](./se-resnet-101/se-resnet-101.md)   | 78.252%/94.206% | 15.239 | 49.274  |
| SE-ResNet 152               | Caffe\*                            | [se-resnet-152](./se-resnet-152/se-resnet-152.md) | 78.506%/94.45% | 22.709 | 66.746 |
| SE-ResNeXt 50               | Caffe\*                            | [se-resnext-50](./se-resnext-50/se-resnext-50.md) | 78.968%/94.63% | 8.533 | 27.526|
| SE-ResNeXt 101              | Caffe\*                            | [se-resnext-101](./se-resnext-101/se-resnext-101.md) | 80.168%/95.19% | 16.054 | 48.886 |
| SqueezeNet v1.0             | Caffe\*                            | [squeezenet1.0](./squeezenet1.0/squeezenet1.0.md)| 57.684%/80.38%| 1.737 | 1.248 |
| SqueezeNet v1.1             | Caffe\*<br>Caffe2\*                | [squeezenet1.1](./squeezenet1.1/squeezenet1.1.md)<br>[squeezenet1.1-caffe2](./squeezenet1.1-caffe2/squeezenet1.1-caffe2.md)| 58.382%/81%<br>56.502%/79.576% | 0.785 | 1.236 |
| VGG 16                      | Caffe\*                            | [vgg16](./vgg16/vgg16.md) | 70.968%/89.878% | 30.974 | 138.358 |
| VGG 19                      | Caffe\*<br>Caffe2\*                | [vgg19](./vgg19/vgg19.md)<br>[vgg19-caffe2](./vgg19-caffe2/vgg19-caffe2.md) | 71.062%/89.832%<br>71.062%/89.832% | 39.3 | 143.667  |

## Segmentation

Semantic segmentation is an extension of object detection problem. Instead of
returning bounding boxes, semantic segmentation models return a "painted"
version of the input image, where the "color" of each pixel represents a certain
class. These networks are much bigger than respective object detection networks,
but they provide a better (pixel-level) localization of objects and they can
detect areas with complex shape.

### Semantic Segmentation

| Model Name                | Implementation | OMZ Model Name | Accuracy | GFlops | mParams |
| ------------------------- | -------------- | -------------- | -------- | ------ | ------- |
| DeepLab V3                | TensorFlow\*   | [deeplabv3](./deeplabv3/deeplabv3.md) | 66.85% | 11.469 | 23.819 |

### Instance Segmentation

Instance segmentation is an extension of object detection and semantic
segmentation problems. Instead of predicting a bounding box around each object
instance instance segmentation model outputs pixel-wise masks for all instances.

| Model Name                     | Implementation | OMZ Model Name | Accuracy | GFlops | mParams |
| ------------------------------ | -------------- | -------------- | -------- | ------ | ------- |
| Mask R-CNN Inception ResNet V2 | TensorFlow\*   | [mask_rcnn_inception_resnet_v2_atrous_coco](./mask_rcnn_inception_resnet_v2_atrous_coco/mask_rcnn_inception_resnet_v2_atrous_coco.md) | 39.8619%/35.3628% | 675.314 | 92.368 |
| Mask R-CNN Inception V2        | TensorFlow\*   | [mask_rcnn_inception_v2_coco](./mask_rcnn_inception_v2_coco/mask_rcnn_inception_v2_coco.md) | 27.1199%/21.4805% | 54.926 | 21.772 |
| Mask R-CNN ResNet 50           | TensorFlow\*   | [mask_rcnn_resnet50_atrous_coco](./mask_rcnn_resnet50_atrous_coco/mask_rcnn_resnet50_atrous_coco.md)| 	29.7512%/27.4597% | 294.738 | 50.222 |
| Mask R-CNN ResNet 101          | TensorFlow\*   | [mask_rcnn_resnet101_atrous_coco](./mask_rcnn_resnet101_atrous_coco/mask_rcnn_resnet101_atrous_coco.md) | 34.9191%/31.301% | 674.58 | 69.188 |

### 3D Semantic Segmentation

| Model Name                | Implementation | OMZ Model Name | Accuracy | GFlops | mParams |
| ------------------------- | -------------- | -------------- | -------- | ------ | ------- |
| Brain Tumor Segmentation  | MXNet\*        | [brain-tumor-segmentation-0001](./brain-tumor-segmentation-0001/brain-tumor-segmentation-0001.md) | 92.4003% | 409.996 | 38.192 |
| Brain Tumor Segmentation 2| PyTorch\*      | [brain-tumor-segmentation-0002](./brain-tumor-segmentation-0002/brain-tumor-segmentation-0002.md) | 91.4826% | 300.801 | 4.51  |

## Object Detection

Several detection models can be used to detect a set of the most popular
objects - for example, faces, people, vehicles. Most of the networks are
SSD-based and provide reasonable accuracy/performance trade-offs.

| Model Name                           | Implementation           | OMZ Model Name | Accuracy | GFlops | mParams |
| ------------------------------------ | ------------------------ | -------------- | -------- | ------ | ------- |
| CTPN                                 | TensorFlow\*             | [ctpn](./ctpn/ctpn.md) | 73.67% | 55.813 | 17.237 |
| CenterNet (CTDET with DLAV0) 384x384 | ONNX\*                   | [ctdet_coco_dlav0_384](./ctdet_coco_dlav0_384/ctdet_coco_dlav0_384.md)| 41.6105%| 34.994 | 17.911 |
| CenterNet (CTDET with DLAV0) 512x512 | ONNX\*                   | [ctdet_coco_dlav0_512](./ctdet_coco_dlav0_512/ctdet_coco_dlav0_512.md)| 44.2756%| 62.211 | 17.911 |
| FaceBoxes                            | PyTorch\*                | [faceboxes-pytorch](./faceboxes-pytorch/faceboxes-pytorch.md)|83.565% | 1.8975 | 1.0059 |
| Faster R-CNN with Inception-ResNet v2| TensorFlow\*             | [faster_rcnn_inception_resnet_v2_atrous_coco](./faster_rcnn_inception_resnet_v2_atrous_coco/faster_rcnn_inception_resnet_v2_atrous_coco.md)| 36.76%/52.41% | 30.687 | 13.307 |
| Faster R-CNN with Inception v2       | TensorFlow\*             | [faster_rcnn_inception_v2_coco](./faster_rcnn_inception_v2_coco/faster_rcnn_inception_v2_coco.md) | 25.65%/40.04%| 30.687 | 13.307 |
| Faster R-CNN with ResNet 50          | TensorFlow\*             | [faster_rcnn_resnet50_coco](./faster_rcnn_resnet50_coco/faster_rcnn_resnet50_coco.md) | 27.47%/42.87% | 57.203 | 29.162 |
| Faster R-CNN with ResNet 101         | TensorFlow\*             | [faster_rcnn_resnet101_coco](./faster_rcnn_resnet101_coco/faster_rcnn_resnet101_coco.md) | 30.95%/47.21%	 | 112.052 | 48.128 |
| MobileFace Detection V1              | MXNet\*                  | [mobilefacedet-v1-mxnet](./mobilefacedet-v1-mxnet/mobilefacedet-v1-mxnet.md)| 	78.7488%| 3.5456 | 7.6828 |
| MTCNN                                | Caffe\*                  | [mtcnn](./mtcnn/mtcnn.md):<br>mtcnn-p <br>mtcnn-r <br>mtcnn-o| 48.1308%/62.2625% | <br>3.3715<br>0.0031<br>0.0263|<br>0.0066<br>0.1002<br>0.3890|
| Pelee                                | Caffe\*                  | [pelee-coco](./pelee-coco/pelee-coco.md) | 21.9761% | 1.290 | 5.98 |
| RetinaNet with Resnet 50             | TensorFlow\*             | [retinanet-tf](./retinanet-tf/retinanet-tf.md) | 33.15% | 238.9469 | 64.9706 |
| R-FCN with Resnet-101                | TensorFlow\*             | [rfcn-resnet101-coco-tf](./rfcn-resnet101-coco-tf/rfcn-resnet101-coco-tf.md) | 28.40%/45.02% | 53.462 | 171.85 |
| SSD 300                              | Caffe\*                  | [ssd300](./ssd300/ssd300.md)  | 85.0791% | 62.815 | 26.285 |
| SSD 512                              | Caffe\*                  | [ssd512](./ssd512/ssd512.md) | 90.3845% | 180.611 | 27.189 |
| SSD with MobileNet                   | Caffe\* <br>TensorFlow\* | [mobilenet-ssd](./mobilenet-ssd/mobilenet-ssd.md) <br>[ssd_mobilenet_v1_coco](./ssd_mobilenet_v1_coco/ssd_mobilenet_v1_coco.md) | 79.8377%<br>23.32%| 2.316~2.494 | 5.783~6.807 |
| SSD with MobileNet FPN               | TensorFlow\*             | [ssd_mobilenet_v1_fpn_coco](./ssd_mobilenet_v1_fpn_coco/ssd_mobilenet_v1_fpn_coco.md) | 35.5453% | 123.309 | 36.188 |
| SSD with MobileNet V2                | TensorFlow\*             | [ssd_mobilenet_v2_coco](./ssd_mobilenet_v2_coco/ssd_mobilenet_v2_coco.md) | 24.9452% | 3.775 | 16.818 |
| SSD lite with MobileNet V2           | TensorFlow\*             | [ssdlite_mobilenet_v2](./ssdlite_mobilenet_v2/ssdlite_mobilenet_v2.md) | 24.2946% | 1.525 | 4.475 |
| SSD with ResNet-50 V1 FPN            | TensorFlow\*             | [ssd_resnet50_v1_fpn_coco](./ssd_resnet50_v1_fpn_coco/ssd_resnet50_v1_fpn_coco.md)| 38.4557% | 178.6807 | 59.9326 |
| SSD with ResNet 34 1200x1200         | PyTorch\*                | [ssd-resnet34-1200-onnx](./ssd-resnet34-1200-onnx/ssd-resnet34-1200-onnx.md) | 20.7198%/39.2752% | 433.411 | 20.058  |
| RetinaFace-R50                       | MXNet\*                  | [retinaface-resnet50](./retinaface-resnet50/retinaface-resnet50.md) | 87.2902% | 100.8478 | 29.427 |
| RetinaFace-Anti-Cov                  | MXNet\*                  | [retinaface-anti-cov](./retinaface-anti-cov/retinaface-anti-cov.md)  | 77.1531% | 2.7781 | 0.5955 |
| YOLO v1 Tiny                         | TensorFlow.js\*          | [yolo-v1-tiny-tf](./yolo-v1-tiny-tf/yolo-v1-tiny-tf.md) | 72.1716% | 6.9883	 |	15.8587 |
| YOLO v2 Tiny                         | TensorFlow.js\*          | [yolo-v2-tiny-tf](./yolo-v2-tiny-tf/yolo-v2-tiny-tf.md) | 27.3443%/29.1184%| 5.4236	 |	11.2295 |
| YOLO v2                              | Keras\*                  | [yolo-v2-tf](./yolo-v2-tf/yolo-v2-tf.md) | 53.1453%/56.483% | 63.0301	 |	50.9526 |
| YOLO v3                              | Keras\*                  | [yolo-v3-tf](./yolo-v3-tf/yolo-v3-tf.md) | 62.2759%/67.7221% | 65.9843	 |	61.9221 |

## Face Recognition

| Model Name                           | Implementation | OMZ Model Name | Accuracy | GFlops | mParams |
| ------------------------------------ | -------------- | -------------- | -------- |------ | ------- |
| FaceNet                              | TensorFlow\*   | [facenet-20180408-102900](./facenet-20180408-102900/facenet-20180408-102900.md) | 98.4522% | 2.846 | 23.469 |
| LResNet34E-IR,ArcFace@ms1m-refine-v1 | MXNet\*        | [face-recognition-resnet34-arcface](./face-recognition-resnet34-arcface/face-recognition-resnet34-arcface.md) | 	98.7488%| 8.934 | 34.129 |
| LResNet50E-IR,ArcFace@ms1m-refine-v1 | MXNet\*        | [face-recognition-resnet50-arcface](./face-recognition-resnet50-arcface/face-recognition-resnet50-arcface.md) | 98.8835%| 12.637 | 43.576 |
| LResNet100E-IR,ArcFace@ms1m-refine-v2| MXNet\*        | [face-recognition-resnet100-arcface](./face-recognition-resnet100-arcface/face-recognition-resnet100-arcface.md) | 99.0218%| 24.209 | 65.131 |
| MobileFaceNet,ArcFace@ms1m-refine-v1 | MXNet\*        | [face-recognition-mobilefacenet-arcface](./face-recognition-mobilefacenet-arcface/face-recognition-mobilefacenet-arcface.md) | 98.8695% | 0.449 | 0.993 |
| SphereFace                           | Caffe\*        | [Sphereface](./Sphereface/Sphereface.md) | 98.8321% | 3.504 | 22.671 |

## Human Pose Estimation

Human pose estimation task is to predict a pose: body skeleton, which consists
of keypoints and connections between them, for every person in an input image or
video. Keypoints are body joints, i.e. ears, eyes, nose, shoulders, knees, etc.
There are two major groups of such metods: top-down and bottom-up.  The first
detects persons in a given frame, crops or rescales detections, then runs pose
estimation network for every detection. These methods are very accurate. The
second finds all keypoints in a given frame, then groups them by person
instances, thus faster than previous, because network runs once.

| Model Name                       | Implementation | OMZ Model Name                | Accuracy | GFlops | mParams |
|--------------------------------- | ---------------| ----------------------------- | -------- | ------ | ------- |
| human-pose-estimation-3d-0001    | PyTorch\*      | [human-pose-estimation-3d-0001](./human-pose-estimation-3d-0001/description/human-pose-estimation-3d-0001.md) | 100.44437mm | 18.998 |  5.074  |
| single-human-pose-estimation-0001| PyTorch\*      | [single-human-pose-estimation-0001](./single-human-pose-estimation-0001/description/single-human-pose-estimation-0001.md) | 69.0491% | 60.125 | 33.165 |

## Monocular Depth Estimation

The task of monocular depth estimation is to predict a depth (or inverse depth) map based on a single input image.
Since this task contains - in the general setting - some ambiguity, the resulting depth maps are often only defined up to an unknown scaling factor.

| Model Name                  | Implementation | OMZ Model Name                | Accuracy | GFlops    | mParams |
| --------------------------- | -------------- | ----------------------------- | -------- | --------- | ------- |
| midasnet                    | PyTorch\*      | [midasnet](./midasnet/midasnet.md)| 7.5878| 207.4915  |    104.0814     |

## Image Inpainting

Image inpainting task is to estimate suitable pixel information to fill holes in images.

| Model Name                | Implementation | OMZ Model Name | Accuracy | GFlops | mParams |
| ------------------------- | ---------------| -------------- | -------- | ------ | ------- |
| GMCNN Inpainting          | TensorFlow\*   | [gmcnn-places2-tf](./gmcnn-places2-tf/gmcnn-places2-tf.md) | 33.47Db | 691.1589 | 12.7773|

## Style Transfer

Style transfer task is to transfer the style of one image to another.

| Model Name                     | Implementation | OMZ Model Name | Accuracy | GFlops | mParams |
| ------------------------------ | ---------------| -------------- | -------- | ------ | ------- |
| fast-neural-style-mosaic-onnx  | ONNX\*         | [fast-neural-style-mosaic-onnx](./fast-neural-style-mosaic-onnx/fast-neural-style-mosaic-onnx.md) | 12.04dB | 15.518 | 1.679 |

## Action Recognition

The task of action recognition is to predict action that is being performed on a short video clip
(tensor formed by stacking sampled frames from input video).

| Model Name                        | Implementation | OMZ Model Name | Accuracy | GFlops | mParams |
| --------------------------------- | ---------------| -------------- | -------- | ------ | ------- |
| RGB-I3D, pretrained on ImageNet\* | TensorFlow\*   | [i3d-rgb-tf](./i3d-rgb-tf/i3d-rgb-tf.md) | 65.96%/86.01% | 278.9815 | 12.6900|

## Colorization

Colorization task is to predict colors of scene from grayscale image.

| Model Name                | Implementation | OMZ Model Name | Accuracy | GFlops | mParams |
| ------------------------- | ---------------| -------------- | -------- | ------ | ------- |
| colorization-v2           | Caffe*         | [colorization-v2](./colorization-v2/colorization-v2.md) | | 64.0047 |  32.2354 |
| colorization-v2-norebal   | Caffe*         | [colorization-v2-norebal](./colorization-v2-norebal/colorization-v2-norebal.md) | | 64.0047 |  32.2354 |

## Sound Classification

The task of sound classification is to predict what sounds are in an audio fragment.

| Model Name                | Implementation | OMZ Model Name | GFlops | mParams |
| ------------------------- | ---------------| -------------- | ------ | ------- |
| ACLNet                    | [PyTorch\*](./aclnet/aclnet.md) | aclnet | 1.4     | 2.7     |

## Legal Information

[*] Other names and brands may be claimed as the property of others.
