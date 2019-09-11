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
| DenseNet 121      | [Caffe\*](./densenet-121/densenet-121.md)<br>[TensorFlow\*](./densenet-121-tf/densenet-121-tf.md)  | densenet-121<br>densenet-121-tf | | 5.289~5.724    | 7.971 |
| DenseNet 161      | [Caffe\*](./densenet-161/densenet-161.md)<br> [TensorFlow\*](./densenet-161-tf/densenet-161-tf.md) | densenet-161<br>densenet-161-tf | | 14.128~15.561  | 28.666 |
| DenseNet 169      | [Caffe\*](./densenet-169/densenet-169.md)<br>[TensorFlow\*](./densenet-169-tf/densenet-169-tf.md)  | densenet-169<br>densenet-169-tf | | 6.16~6.788 | 14.139 |
| DenseNet 201      | [Caffe\*](./densenet-201/densenet-201.md) | densenet-201 | | 8.673  | 20.001  |
| Inception (GoogleNet) V1 | [Caffe\*](./googlenet-v1/googlenet-v1.md) | googlenet-v1 | | 3.266 | 6.999 |
| Inception (GoogleNet) V2 | [Caffe\*](./googlenet-v2/googlenet-v2.md) | googlenet-v2 | | 4.058 | 11.185 |
| Inception (GoogleNet) V3 | [Caffe\*](./googlenet-v3/googlenet-v3.md)<br>[PyTorch\*](./googlenet-v3-pytorch/googlenet-v3-pytorch.md) | googlenet-v3 <br> googlenet-v3-pytorch |  | 11.469 | 23.817 |
| Inception (GoogleNet) V4 | [Caffe\*](./googlenet-v4/googlenet-v4.md) | googlenet-v4| | 24.584 | 42.648 |
| Inception-ResNet V2      | [Caffe\*](./inception-resnet-v2/inception-resnet-v2.md)<br>[TensorFlow\*](./inception-resnet-v2-tf/inception-resnet-v2-tf.md) | inception-resnet-v2 <br>  inception-resnet-v2-tf | | 22.227~26.405 | 30.223~55.813 |
| MobileNet V1 0.25 128 | [Caffe\*](./mobilenet-v1-0.25-128/mobilenet-v1-0.25-128.md) | mobilenet-v1-0.25-128 | | 0.028 | 0.468 | 
| MobileNet V1 0.5 160 | [Caffe\*](./mobilenet-v1-0.50-160/mobilenet-v1-0.50-160.md)  | mobilenet-v1-0.50-160 | | 0.156 | 1.327 |
| MobileNet V1 0.5 224 | [Caffe\*](./mobilenet-v1-0.50-224/mobilenet-v1-0.50-224.md)  | mobilenet-v1-0.50-224 | | 0.304 | 1.327 |
| MobileNet V1 1.0 224 | [Caffe\*](./mobilenet-v1-1.0-224/mobilenet-v1-1.0-224.md)<br>[TensorFlow\*](./mobilenet-v1-1.0-224-tf/mobilenet-v1-1.0-224-tf.md) | mobilenet-v1-1.0-224 <br> mobilenet-v1-1.0-224-tf | | 1.148 | 4.221 |
| MobileNet V2 1.0 224 | [Caffe\*](./mobilenet-v2/mobilenet-v2.md)<br>[TensorFlow\*](./mobilenet-v2-1.0-224/mobilenet-v2-1.0-224.md)<br>[PyTorch\*](./mobilenet-v2-pytorch/mobilenet-v2-pytorch.md) | mobilenet-v2 <br>  mobilenet-v2-1.0-224 <br> mobilenet-v2-pytorch | | 0.615~0.876 | 3.489 |
| MobileNet V2 1.4 224 | [TensorFlow\*](./mobilenet-v2-1.4-224/mobilenet-v2-1.4-224.md)  | mobilenet-v2-1.4-224 | | 1.183  | 6.087 |
| ResNet 50            | [Caffe\*](./resnet-50/resnet-50.md) <br> [PyTorch\*](./resnet-50-pytorch/resnet-50-pytorch.md) | resnet-50 <br> resnet-50-pytorch | | 6.996~8.216 | 25.53 |
| ResNet 101           | [Caffe\*](./resnet-101/resnet-101.md) | resnet-101 | | 14.441 | 44.496 |
| ResNet 152           | [Caffe\*](./resnet-152/resnet-152.md) | resnet-152 | | 21.89 | 60.117 |
| SE-Inception         | [Caffe\*](./se-inception/se-inception.md)      | se-inception | | 4.091 | 11.922 |
| SE-ResNet 50         | [Caffe\*](./se-resnet-50/se-resnet-50.md)      | se-resnet-50 | | 7.775 | 28.061 |
| SE-ResNet 101        | [Caffe\*](./se-resnet-101/se-resnet-101.md)    | se-resnet-101  | | 15.239 | 49.274  |
| SE-ResNet 152        | [Caffe\*](./se-resnet-152/se-resnet-152.md)    | se-resnet-152 | | 22.709 | 66.746 |
| SE-ResNeXt 50        | [Caffe\*](./se-resnext-50/se-resnext-50.md)    | se-resnext-50 | | 8.533 | 27.526|
| SE-ResNeXt 101       | [Caffe\*](./se-resnext-101/se-resnext-101.md)  | se-resnext-101 | | 16.054 | 48.886 |
| SqueezeNet v1.0      | [Caffe\*](./squeezenet1.0/squeezenet1.0.md)    | squeezenet1.0| | 1.737 | 1.248 |
| SqueezeNet v1.1      | [Caffe\*](./squeezenet1.1/squeezenet1.1.md)    | squeezenet1.1| | 0.785 | 1.236 |
| VGG 16               | [Caffe\*](./vgg16/vgg16.md) | vgg16 | | 30.974 | 138.358 |
| VGG 19               | [Caffe\*](./vgg19/vgg19.md) | vgg19 | | 39.3 | 143.667  |

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

## Object Detection

Several detection models can be used to detect a set of the most popular
objects - for example, faces, people, vehicles. Most of the networks are
SSD-based and provide reasonable accuracy/performance trade-offs.

| Model Name                | Implementation | OMZ Model Name | GFlops | mParams |
| ------------------------- | -------------- | -------------- | ------ | ------- |
| CTPN                      | [TensorFlow\*](./ctpn/ctpn.md) | ctpn | 55.813 | 17.237 | |
| Faster R-CNN with Inception-ResNet v2 | [TensorFlow\*](./faster_rcnn_inception_resnet_v2_atrous_coco/faster_rcnn_inception_resnet_v2_atrous_coco.md) | faster_rcnn_inception_resnet_v2_atrous_coco| 30.687 | 13.307 |
| Faster R-CNN with Inception v2 | [TensorFlow\*](./faster_rcnn_inception_v2_coco/faster_rcnn_inception_v2_coco.md) | faster_rcnn_inception_v2_coco | 30.687 | 13.307 |
| Faster R-CNN with ResNet 50    | [TensorFlow\*](./faster_rcnn_resnet50_coco/faster_rcnn_resnet50_coco.md) | faster_rcnn_resnet50_coco | 57.203 | 29.162 |
| Faster R-CNN with ResNet 101   | [TensorFlow\*](./faster_rcnn_resnet101_coco/faster_rcnn_resnet101_coco.md) |faster_rcnn_resnet101_coco | 112.052 | 48.128 |
| MTCNN                     | Caffe\*: <br>[proposal](./mtcnn-p/mtcnn-p.md) <br>[refine](./mtcnn-r/mtcnn-r.md) <br> [output](./mtcnn-o/mtcnn-o.md) | <br>mtcnn-p <br>mtcnn-r <br>mtcnn-o| | |
| SSD 300                   | [Caffe\*](./ssd300/ssd300.md) | ssd300 | 62.815 | 26.285 |
| SSD 512                   | [Caffe\*](./ssd512/ssd512.md) | ssd512 | 180.611 | 27.189 |
| SSD with MobileNet        | [Caffe\*](./mobilenet-ssd/mobilenet-ssd.md) <br>[TensorFlow\*](./ssd_mobilenet_v1_coco/ssd_mobilenet_v1_coco.md) | mobilenet-ssd <br>ssd_mobilenet_v1_coco | 2.316~2.494 | 5.783~6.807 |
| SSD with MobileNet FPN    | [TensorFlow\*](./ssd_mobilenet_v1_fpn_coco/ssd_mobilenet_v1_fpn_coco.md) | ssd_mobilenet_v1_fpn_coco | 123.309 | 36.188 |
| SSD with MobileNet V2     | [TensorFlow\*](./ssd_mobilenet_v2_coco/ssd_mobilenet_v2_coco.md) | ssd_mobilenet_v2_coco | 3.775 | 16.818 |
| SSD lite with MobileNet V2 | [TensorFlow\*](./ssdlite_mobilenet_v2/ssdlite_mobilenet_v2.md)  | ssdlite_mobilenet_v2 | 1.525 | 4.475 |

## Face Recognition

| Model Name                | Implementation | OMZ Model Name | GFlops | mParams |
| ------------------------- | -------------- | -------------- |------ | ------- |
| FaceNet                   | [TensorFlow\*](./facenet-20180408-102900/facenet-20180408-102900.md) | facenet-20180408-102900 | 2.846 | 23.469 |
| LResNet34E-IR,ArcFace@ms1m-refine-v1 | [MXNet\*](./face-recognition-resnet34-arcface/face-recognition-resnet34-arcface.md) | face-recognition-resnet34-arcface | 8.934 | 34.129 |
| LResNet50E-IR,ArcFace@ms1m-refine-v1 | [MXNet\*](./face-recognition-resnet50-arcface/face-recognition-resnet50-arcface.md) | face-recognition-resnet50-arcface | 12.637 | 43.576 |
| LResNet100E-IR,ArcFace@ms1m-refine-v2 | [MXNet\*](./face-recognition-resnet100-arcface/face-recognition-resnet100-arcface.md) | face-recognition-resnet100-arcface | 24.209 | 65.131 |
| MobileFaceNet,ArcFace@ms1m-refine-v1 | [MXNet\*](./face-recognition-mobilefacenet-arcface/face-recognition-mobilefacenet-arcface.md) | face-recognition-mobilefacenet-arcface | 0.449 | 0.993 |
| SphereFace                | [Caffe\*](./Sphereface/Sphereface.md) | Sphereface | 3.504 | 22.671 |

## Legal Information

[*] Other names and brands may be claimed as the property of others.
