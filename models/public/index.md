# Overview of OpenVINO&trade; Toolkit Public Pre-Trained Models

OpenVINO&trade; toolkit provides a set of public pre-trained models
that you can use for learning and demo purposes or for developing deep learning
software. Most recent version is available in the [repo on Github](https://github.com/openvinotoolkit/open_model_zoo).
The table [Public Pre-Trained Models Device Support](./device_support.md) summarizes devices supported by each model.

You can download models and convert them into Inference Engine format (\*.xml + \*.bin) using the OpenVINO™ [Model Downloader](../../tools/downloader/README.md) and other automation tools.

## Classification

| Model Name                  | Implementation                     | OMZ Model Name | Accuracy | GFlops | mParams |
| --------------------------- | -----------------------------------| -------------- | -------- | ------ | ------- |
| AlexNet                     | Caffe\*                            | [alexnet](./alexnet/README.md)   | 56.598%/79.812% | 1.5 | 60.965 |
| AntiSpoofNet                | PyTorch\*                          | [anti-spoof-mn3](./anti-spoof-mn3/README.md) | 3.81% | 0.15 | 3.02 |
| CaffeNet                    | Caffe\*                            | [caffenet](./caffenet/README.md)  | 56.714%/79.916% | 1.5 | 60.965 |
| DenseNet 121                | Caffe\*<br>TensorFlow\*<br>Caffe2\*| [densenet-121](./densenet-121/README.md)<br>[densenet-121-tf](./densenet-121-tf/README.md)<br>[densenet-121-caffe2](./densenet-121-caffe2/README.md) | 74.42%/92.136%<br>74.46%/92.13% <br>74.904%/92.192% | 5.723~5.7287 | 7.971 |
| DenseNet 161                | Caffe\*<br>TensorFlow\*            | [densenet-161](./densenet-161/README.md)<br>[densenet-161-tf](./densenet-161-tf/README.md) | 77.55%/93.92%<br>76.446%/93.228%| 14.128~15.561  | 28.666 |
| DenseNet 169                | Caffe\*<br>TensorFlow\*            | [densenet-169](./densenet-169/README.md)<br>[densenet-169-tf](./densenet-169-tf/README.md) | 76.106%/93.106%<br>76.14%/93.12%| 6.788~6.7932 | 14.139 |
| DenseNet 201                | Caffe\*<br>TensorFlow\*            | [densenet-201](./densenet-201/README.md)<br>[densenet-201-tf](./densenet-201-tf/README.md)| 76.886%/93.556%<br>76.93%/93.56% | 8.673~8.6786  | 20.001  |
| DLA 34                      | PyTorch\*                          | [dla-34](./dla-34/README.md) | 74.64%/92.06% | 6.1368 | 15.7344 |
| EfficientNet B0             | TensorFlow\*<br>PyTorch\*          | [efficientnet-b0](./efficientnet-b0/README.md)<br>[efficientnet-b0-pytorch](./efficientnet-b0-pytorch/README.md) | 75.70%/92.76%<br>76.91%/93.21% | 0.819 | 5.268 |
| EfficientNet B0 AutoAugment | TensorFlow\*                       | [efficientnet-b0_auto_aug](./efficientnet-b0_auto_aug/README.md) | 76.43%/93.04% | 0.819 | 5.268 |
| EfficientNet B5             | TensorFlow\*<br>PyTorch\*          | [efficientnet-b5](./efficientnet-b5/README.md)<br>[efficientnet-b5-pytorch](./efficientnet-b5-pytorch/README.md) | 83.33%/96.67%<br>83.69%/96.71% | 21.252 | 30.303 |
| EfficientNet B7             | PyTorch\*                          | [efficientnet-b7-pytorch](./efficientnet-b7-pytorch/README.md) | 84.42%/96.91% | 77.618  | 66.193 |
| EfficientNet B7 AutoAugment | TensorFlow\*                       | [efficientnet-b7_auto_aug](./efficientnet-b7_auto_aug/README.md) | 84.68%/97.09% | 77.618  | 66.193 |
| HBONet 1.0                  | PyTorch\*                          | [hbonet-1.0](./hbonet-1.0/README.md)  | 73.1%/91.0% | 0.6208 | 4.5443 |
| HBONet 0.5                  | PyTorch\*                          | [hbonet-0.5](./hbonet-0.5/README.md)  | 67.0%/86.9% | 0.1977 | 2.5287 |
| HBONet 0.25                 | PyTorch\*                          | [hbonet-0.25](./hbonet-0.25/README.md) | 57.3%/79.8% | 0.0758 | 1.9299 |
| Inception (GoogleNet) V1    | Caffe\*<br>TensorFlow\*            | [googlenet-v1](./googlenet-v1/README.md)<br>[googlenet-v1-tf](./googlenet-v1-tf/README.md) | 68.928%/89.144%<br>69.814%/89.6% | 3.016~3.266 | 6.619~6.999 |
| Inception (GoogleNet) V2    | Caffe\*<br>TensorFlow\*            | [googlenet-v2](./googlenet-v2/README.md)<br>[googlenet-v2-tf](./googlenet-v2-tf/README.md) | 72.024%/90.844%<br>74.084%/91.798%| 4.058 | 11.185 |
| Inception (GoogleNet) V3    | TensorFlow\*<br>PyTorch\*          | [googlenet-v3](./googlenet-v3/README.md) <br> [googlenet-v3-pytorch](./googlenet-v3-pytorch/README.md) | 77.904%/93.808%<br>77.69%/93.7% | 11.469 | 23.817 |
| Inception (GoogleNet) V4    | TensorFlow\*                       | [googlenet-v4-tf](./googlenet-v4-tf/README.md) | 80.204%/95.21% | 24.584 | 42.648 |
| Inception-ResNet V2         | TensorFlow\*                       | [inception-resnet-v2-tf](./inception-resnet-v2-tf/README.md) | 80.14%/95.10% | 22.227 | 30.223 |
| MixNet L                    | TensorFlow\*                       | [mixnet-l](./mixnet-l/README.md)  | 78.30%/93.91% | 0.565 | 7.3 |
| MobileNet V1 0.25 128       | Caffe\*                            | [mobilenet-v1-0.25-128](./mobilenet-v1-0.25-128/README.md)  | 40.54%/65% | 0.028 | 0.468 |
| MobileNet V1 0.5 160        | Caffe\*                            | [mobilenet-v1-0.50-160](./mobilenet-v1-0.50-160/README.md) | 59.86%/82.04% | 0.156 | 1.327 |
| MobileNet V1 0.5 224        | Caffe\*                            | [mobilenet-v1-0.50-224](./mobilenet-v1-0.50-224/README.md) | 63.042%/84.934%| 0.304 | 1.327 |
| MobileNet V1 1.0 224        | Caffe\*<br>TensorFlow\*            | [mobilenet-v1-1.0-224](./mobilenet-v1-1.0-224/README.md)<br>[mobilenet-v1-1.0-224-tf](./mobilenet-v1-1.0-224-tf/README.md)| 69.496%/89.224%<br>71.03%/89.94% | 1.148 | 4.221 |
| MobileNet V2 1.0 224        | Caffe\*<br>TensorFlow\*<br>PyTorch\*| [mobilenet-v2](./mobilenet-v2/README.md) <br>[mobilenet-v2-1.0-224](./mobilenet-v2-1.0-224/README.md)<br>[mobilenet-v2-pytorch](./mobilenet-v2-pytorch/README.md) | 71.218%/90.178%<br>71.85%/90.69%<br>71.81%/90.396% | 0.615~0.876 | 3.489 |
| MobileNet V2 1.4 224        | TensorFlow\*                       | [mobilenet-v2-1.4-224](./mobilenet-v2-1.4-224/README.md) | 74.09%/91.97% | 1.183  | 6.087 |
| MobileNet V3 Small 1.0      | TensorFlow\*                       | [mobilenet-v3-small-1.0-224-tf](./mobilenet-v3-small-1.0-224-tf/README.md) | 67.36%/87.45% | 0.121 | 2.537 |
| MobileNet V3 Large 1.0      | TensorFlow\*                       | [mobilenet-v3-large-1.0-224-tf](./mobilenet-v3-large-1.0-224-tf/README.md) | 75.70%/92.76% | 0.4536 | 5.4721 |
| NFNet F0                    | PyTorch\*                          | [nfnet-f0](./nfnet-f0/README.md) | 83.34%/96.56% | 24.8053 | 71.4444 |
| DenseNet 121, alpha=0.125   | MXNet\*                            | [octave-densenet-121-0.125](./octave-densenet-121-0.125/README.md) | 76.066%/93.044% | 4.883 | 7.977 |
| RegNetX-3.2GF               | PyTorch\*                          | [regnetx-3.2gf](./regnetx-3.2gf/README.md) | 78.17%/94.08% | 6.3893 | 15.2653 |
| ResNet 26, alpha=0.25       | MXNet\*                            | [octave-resnet-26-0.25](./octave-resnet-26-0.25/README.md)     | 76.076%/92.584%| 3.768 | 15.99 |
| ResNet 50, alpha=0.125      | MXNet\*                            | [octave-resnet-50-0.125](./octave-resnet-50-0.125/README.md)    | 78.19%/93.862% | 7.221 | 25.551 |
| ResNet 101, alpha=0.125     | MXNet\*                            | [octave-resnet-101-0.125](./octave-resnet-101-0.125/README.md)   | 79.182%/94.42% | 13.387 | 44.543 |
| ResNet 200, alpha=0.125     | MXNet\*                            | [octave-resnet-200-0.125](./octave-resnet-200-0.125/README.md)   | 79.99%/94.866%| 25.407 | 64.667 |
| ResNeXt 50, alpha=0.25      | MXNet\*                            | [octave-resnext-50-0.25](./octave-resnext-50-0.25/README.md)    | 78.772%/94.18% | 6.444 | 25.02 |
| ResNeXt 101, alpha=0.25     | MXNet\*                            | [octave-resnext-101-0.25](./octave-resnext-101-0.25/README.md)   | 79.556%/94.444% | 11.521 | 44.169 |
| SE-ResNet 50, alpha=0.125   | MXNet\*                            | [octave-se-resnet-50-0.125](./octave-se-resnet-50-0.125/README.md) | 78.706%/94.09% | 7.246 | 28.082 |
| open-closed-eye-0001        | PyTorch\*                          | [open-closed-eye-0001](./open-closed-eye-0001/README.md) |  95.84%   | 0.0014 | 0.0113|
| RepVGG A0                   | PyTorch\*                          | [repvgg-a0](./repvgg-a0/README.md) | 72.40%/90.49% | 2.7286 | 8.3094 |
| RepVGG B1                   | PyTorch\*                          | [repvgg-b1](./repvgg-b1/README.md) | 78.37%/94.09% | 23.6472 | 51.8295|
| RepVGG B3                   | PyTorch\*                          | [repvgg-b3](./repvgg-b3/README.md) | 80.50%/95.25% | 52.4407 | 110.9609|
| ResNeSt 50                  | PyTorch\*                          | [resnest-50-pytorch](./resnest-50-pytorch/README.md) | 81.11%/95.36% | 10.8148 |  27.4493|
| ResNet 18                   | PyTorch\*                          | [resnet-18-pytorch](./resnet-18-pytorch/README.md) | 69.754%/89.088% | 3.637 |  11.68 |
| ResNet 34                   | PyTorch\*                          | [resnet-34-pytorch](./resnet-34-pytorch/README.md) | 73.30%/91.42% | 7.3409 | 21.7892 |
| ResNet 50                   | PyTorch\*<br>Caffe2\*<br>TensorFlow\*| [resnet-50-pytorch](./resnet-50-pytorch/README.md)<br>[resnet-50-caffe2](./resnet-50-caffe2/README.md)<br>[resnet-50-tf](./resnet-50-tf/README.md)| 75.168%/92.212%<br>76.128%/92.858%<br>76.38%/93.188%<br>76.17%/92.98% | 6.996~8.216 | 25.53 |
| ReXNet V1 x1.0              | PyTorch\*                          | [rexnet-v1-x1.0](./rexnet-v1-x1.0/README.md) | 77.86%/93.87% | 0.8325 | 4.7779 |
| SE-Inception                | Caffe\*                            | [se-inception](./se-inception/README.md) | 75.996%/92.964% | 4.091 | 11.922 |
| SE-ResNet 50                | Caffe\*                            | [se-resnet-50](./se-resnet-50/README.md) | 77.596%/93.85% | 7.775 | 28.061 |
| SE-ResNet 101               | Caffe\*                            | [se-resnet-101](./se-resnet-101/README.md)   | 78.252%/94.206% | 15.239 | 49.274  |
| SE-ResNet 152               | Caffe\*                            | [se-resnet-152](./se-resnet-152/README.md) | 78.506%/94.45% | 22.709 | 66.746 |
| SE-ResNeXt 50               | Caffe\*                            | [se-resnext-50](./se-resnext-50/README.md) | 78.968%/94.63% | 8.533 | 27.526|
| SE-ResNeXt 101              | Caffe\*                            | [se-resnext-101](./se-resnext-101/README.md) | 80.168%/95.19% | 16.054 | 48.886 |
| Shufflenet V2 x1.0          | PyTorch\*                          | [shufflenet-v2-x1.0](./shufflenet-v2-x1.0/README.md) | 69.36%/88.32% | 0.2957 | 2.2705 |
| SqueezeNet v1.0             | Caffe\*                            | [squeezenet1.0](./squeezenet1.0/README.md)| 57.684%/80.38%| 1.737 | 1.248 |
| SqueezeNet v1.1             | Caffe\*<br>Caffe2\*                | [squeezenet1.1](./squeezenet1.1/README.md)<br>[squeezenet1.1-caffe2](./squeezenet1.1-caffe2/README.md)| 58.382%/81%<br>56.502%/79.576% | 0.785 | 1.236 |
| VGG 16                      | Caffe\*                            | [vgg16](./vgg16/README.md) | 70.968%/89.878% | 30.974 | 138.358 |
| VGG 19                      | Caffe\*<br>Caffe2\*                | [vgg19](./vgg19/README.md)<br>[vgg19-caffe2](./vgg19-caffe2/README.md) | 71.062%/89.832%<br>71.062%/89.832% | 39.3 | 143.667  |

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
| DeepLab V3                | TensorFlow\*   | [deeplabv3](./deeplabv3/README.md) | 66.85% | 11.469 | 23.819 |
| HRNet V2 C1 Segmentation  | PyTorch\*      | [hrnet-v2-c1-segmentation](./hrnet-v2-c1-segmentation/README.md) | 77.69% | 81.993 | 66.4768 |
| Fastseg MobileV3Large LR-ASPP, F=128  | PyTorch\*      | [fastseg-large](./fastseg-large/README.md) | 72.67% | 140.9611 | 3.2 |
| Fastseg MobileV3Small LR-ASPP, F=128  | PyTorch\*      | [fastseg-small](./fastseg-small/README.md) | 67.15% | 69.2204 | 1.1 |
| PSPNet R-50-D8  | PyTorch\*      | [pspnet-pytorch](./pspnet-pytorch/README.md) | 70.6% | 357.1719 | 46.5827 |

### Instance Segmentation

Instance segmentation is an extension of object detection and semantic
segmentation problems. Instead of predicting a bounding box around each object
instance instance segmentation model outputs pixel-wise masks for all instances.

| Model Name                     | Implementation | OMZ Model Name | Accuracy | GFlops | mParams |
| ------------------------------ | -------------- | -------------- | -------- | ------ | ------- |
| Mask R-CNN Inception ResNet V2 | TensorFlow\*   | [mask_rcnn_inception_resnet_v2_atrous_coco](./mask_rcnn_inception_resnet_v2_atrous_coco/README.md) | 39.86%/35.36% | 675.314 | 92.368 |
| Mask R-CNN Inception V2        | TensorFlow\*   | [mask_rcnn_inception_v2_coco](./mask_rcnn_inception_v2_coco/README.md) | 27.12%/21.48% | 54.926 | 21.772 |
| Mask R-CNN ResNet 50           | TensorFlow\*   | [mask_rcnn_resnet50_atrous_coco](./mask_rcnn_resnet50_atrous_coco/README.md)| 	29.75%/27.46% | 294.738 | 50.222 |
| Mask R-CNN ResNet 101          | TensorFlow\*   | [mask_rcnn_resnet101_atrous_coco](./mask_rcnn_resnet101_atrous_coco/README.md) | 34.92%/31.30% | 674.58 | 69.188 |
| YOLACT ResNet 50 FPN | PyTorch\* | [yolact-resnet50-fpn-pytorch](./yolact-resnet50-fpn-pytorch/README.md) | 28.0%/30.69% | 118.575 |  36.829  |

### 3D Semantic Segmentation

| Model Name                | Implementation | OMZ Model Name | Accuracy | GFlops | mParams |
| ------------------------- | -------------- | -------------- | -------- | ------ | ------- |
| Brain Tumor Segmentation  | MXNet\*        | [brain-tumor-segmentation-0001](./brain-tumor-segmentation-0001/README.md) | 92.4003% | 409.996 | 38.192 |
| Brain Tumor Segmentation 2| PyTorch\*      | [brain-tumor-segmentation-0002](./brain-tumor-segmentation-0002/README.md) | 91.4826% | 300.801 | 4.51  |

## Object Detection

Several detection models can be used to detect a set of the most popular
objects - for example, faces, people, vehicles. Most of the networks are
SSD-based and provide reasonable accuracy/performance trade-offs.

| Model Name                           | Implementation           | OMZ Model Name | Accuracy | GFlops | mParams |
| ------------------------------------ | ------------------------ | -------------- | -------- | ------ | ------- |
| CTPN                                 | TensorFlow\*             | [ctpn](./ctpn/README.md) | 73.67% | 55.813 | 17.237 |
| CenterNet (CTDET with DLAV0) 384x384 | ONNX\*                   | [ctdet_coco_dlav0_384](./ctdet_coco_dlav0_384/README.md)| 41.6105%| 34.994 | 17.911 |
| CenterNet (CTDET with DLAV0) 512x512 | ONNX\*                   | [ctdet_coco_dlav0_512](./ctdet_coco_dlav0_512/README.md)| 44.2756%| 62.211 | 17.911 |
| EfficientDet-D0                      | TensorFlow\*             | [efficientdet-d0-tf](./efficientdet-d0-tf/README.md)| 31.95% | 2.54 | 3.9 |
| EfficientDet-D1                      | TensorFlow\*             | [efficientdet-d1-tf](./efficientdet-d1-tf/README.md)| 37.54% | 6.1 | 6.6 |
| FaceBoxes                            | PyTorch\*                | [faceboxes-pytorch](./faceboxes-pytorch/README.md)|83.565% | 1.8975 | 1.0059 |
| Face Detection Retail                | Caffe\*                  | [face-detection-retail-0044](./face-detection-retail-0044/README.md) | 83.00% | 1.067 | 0.588 |
| Faster R-CNN with Inception-ResNet v2| TensorFlow\*             | [faster_rcnn_inception_resnet_v2_atrous_coco](./faster_rcnn_inception_resnet_v2_atrous_coco/README.md)| 40.69% | 30.687 | 13.307 |
| Faster R-CNN with Inception v2       | TensorFlow\*             | [faster_rcnn_inception_v2_coco](./faster_rcnn_inception_v2_coco/README.md) | 26.24%| 30.687 | 13.307 |
| Faster R-CNN with ResNet 50          | TensorFlow\*             | [faster_rcnn_resnet50_coco](./faster_rcnn_resnet50_coco/README.md) | 31.09% | 57.203 | 29.162 |
| Faster R-CNN with ResNet 101         | TensorFlow\*             | [faster_rcnn_resnet101_coco](./faster_rcnn_resnet101_coco/README.md) | 35.72% | 112.052 | 48.128 |
| MobileFace Detection V1              | MXNet\*                  | [mobilefacedet-v1-mxnet](./mobilefacedet-v1-mxnet/README.md)| 	78.7488%| 3.5456 | 7.6828 |
| MTCNN                                | Caffe\*                  | [mtcnn](./mtcnn/README.md):<br>mtcnn-p <br>mtcnn-r <br>mtcnn-o| 48.1308%/62.2625% | <br>3.3715<br>0.0031<br>0.0263|<br>0.0066<br>0.1002<br>0.3890|
| Pelee                                | Caffe\*                  | [pelee-coco](./pelee-coco/README.md) | 21.9761% | 1.290 | 5.98 |
| RetinaFace with ResNet 50            | PyTorch\*                | [retinaface-resnet50-pytorch](./retinaface-resnet50-pytorch/README.md) | 91.78% | 88.8627 | 27.2646 |
| RetinaNet with Resnet 50             | TensorFlow\*             | [retinanet-tf](./retinanet-tf/README.md) | 33.15% | 238.9469 | 64.9706 |
| R-FCN with Resnet-101                | TensorFlow\*             | [rfcn-resnet101-coco-tf](./rfcn-resnet101-coco-tf/README.md) | 28.40%/45.02% | 53.462 | 171.85 |
| SSD 300                              | Caffe\*                  | [ssd300](./ssd300/README.md)  | 87.09% | 62.815 | 26.285 |
| SSD 512                              | Caffe\*                  | [ssd512](./ssd512/README.md) | 91.07% | 180.611 | 27.189 |
| SSD with MobileNet                   | Caffe\* <br>TensorFlow\* | [mobilenet-ssd](./mobilenet-ssd/README.md) <br>[ssd_mobilenet_v1_coco](./ssd_mobilenet_v1_coco/README.md) | 67.00%<br>23.32%| 2.316~2.494 | 5.783~6.807 |
| SSD with MobileNet FPN               | TensorFlow\*             | [ssd_mobilenet_v1_fpn_coco](./ssd_mobilenet_v1_fpn_coco/README.md) | 35.5453% | 123.309 | 36.188 |
| SSD with MobileNet V2                | TensorFlow\*             | [ssd_mobilenet_v2_coco](./ssd_mobilenet_v2_coco/README.md) | 24.9452% | 3.775 | 16.818 |
| SSD lite with MobileNet V2           | TensorFlow\*             | [ssdlite_mobilenet_v2](./ssdlite_mobilenet_v2/README.md) | 24.2946% | 1.525 | 4.475 |
| SSD with ResNet-50 V1 FPN            | TensorFlow\*             | [ssd_resnet50_v1_fpn_coco](./ssd_resnet50_v1_fpn_coco/README.md)| 38.4557% | 178.6807 | 59.9326 |
| SSD with ResNet 34 1200x1200         | PyTorch\*                | [ssd-resnet34-1200-onnx](./ssd-resnet34-1200-onnx/README.md) | 20.7198%/39.2752% | 433.411 | 20.058  |
| Ultra Lightweight Face Detection RFB 320| PyTorch\*             | [ultra-lightweight-face-detection-rfb-320](./ultra-lightweight-face-detection-rfb-320/README.md)|84.78% | 0.2106 | 0.3004 |
| Ultra Lightweight Face Detection slim 320| PyTorch\*            | [ultra-lightweight-face-detection-slim-320](./ultra-lightweight-face-detection-slim-320/README.md)|83.32% | 0.1724 | 0.2844 |
| Vehicle License Plate Detection Barrier | TensorFlow\* | [vehicle-license-plate-detection-barrier-0123](./vehicle-license-plate-detection-barrier-0123/README.md) | 	99.52% | 0.271 | 0.547 |
| YOLO v1 Tiny                         | TensorFlow.js\*          | [yolo-v1-tiny-tf](./yolo-v1-tiny-tf/README.md) | 54.79% | 6.9883	 |	15.8587 |
| YOLO v2 Tiny                         | Keras\*                  | [yolo-v2-tiny-tf](./yolo-v2-tiny-tf/README.md) | 27.3443%/29.1184%| 5.4236	 |	11.2295 |
| YOLO v2                              | Keras\*                  | [yolo-v2-tf](./yolo-v2-tf/README.md) | 53.1453%/56.483% | 63.0301	 |	50.9526 |
| YOLO v3                              | Keras\*                  | [yolo-v3-tf](./yolo-v3-tf/README.md) | 62.2759%/67.7221% | 65.9843	 |	61.9221 |
| YOLO v3 Tiny                         | Keras\*                  | [yolo-v3-tiny-tf](./yolo-v3-tiny-tf/README.md) | 35.9%/39.7% | 5.582  | 8.848 |
| YOLO v4                              | Keras\*                  | [yolo-v4-tf](./yolo-v4-tf/README.md) | 71.23%/77.40%/50.26% | 129.5567	 |	64.33 |
| YOLO v4 Tiny                         | Keras\*                  | [yolo-v4-tiny-tf](./yolo-v4-tiny-tf/README.md) | | 6.9289 | 6.0535 |

## Face Recognition

| Model Name                           | Implementation | OMZ Model Name | Accuracy | GFlops | mParams |
| ------------------------------------ | -------------- | -------------- | -------- |------ | ------- |
| FaceNet                              | TensorFlow\*   | [facenet-20180408-102900](./facenet-20180408-102900/README.md) | 99.14% | 2.846 | 23.469 |
| LResNet100E-IR,ArcFace@ms1m-refine-v2| MXNet\*        | [face-recognition-resnet100-arcface-onnx](./face-recognition-resnet100-arcface-onnx/README.md) | 99.68%| 24.2115  | 65.1320 |
| SphereFace                           | Caffe\*        | [Sphereface](./Sphereface/README.md) | 98.8321% | 3.504 | 22.671 |

## Human Pose Estimation

Human pose estimation task is to predict a pose: body skeleton, which consists
of keypoints and connections between them, for every person in an input image or
video. Keypoints are body joints, i.e. ears, eyes, nose, shoulders, knees, etc.
There are two major groups of such methods: top-down and bottom-up.  The first
detects persons in a given frame, crops or rescales detections, then runs pose
estimation network for every detection. These methods are very accurate. The
second finds all keypoints in a given frame, then groups them by person
instances, thus faster than previous, because network runs once.

| Model Name                       | Implementation | OMZ Model Name                | Accuracy | GFlops | mParams |
|--------------------------------- | ---------------| ----------------------------- | -------- | ------ | ------- |
| human-pose-estimation-3d-0001    | PyTorch\*      | [human-pose-estimation-3d-0001](./human-pose-estimation-3d-0001/README.md) | 100.44437mm | 18.998 |  5.074  |
| single-human-pose-estimation-0001| PyTorch\*      | [single-human-pose-estimation-0001](./single-human-pose-estimation-0001/README.md) | 69.0491% | 60.125 | 33.165 |
| higher-hrnet-w32-human-pose-estimation | PyTorch\* | [higher-hrnet-w32-human-pose-estimation](./higher-hrnet-w32-human-pose-estimation/README.md) | 64.64%    | 92.8364 | 28.6180 |

## Monocular Depth Estimation

The task of monocular depth estimation is to predict a depth (or inverse depth) map based on a single input image.
Since this task contains - in the general setting - some ambiguity, the resulting depth maps are often only defined up to an unknown scaling factor.

| Model Name                  | Implementation | OMZ Model Name                | Accuracy | GFlops    | mParams |
| --------------------------- | -------------- | ----------------------------- | -------- | --------- | ------- |
| midasnet                    | PyTorch\*      | [midasnet](./midasnet/README.md)| 0.07071 | 207.25144  |    104.081     |
| FCRN ResNet50-Upproj        | TensorFlow\*   | [fcrn-dp-nyu-depth-v2-tf](./fcrn-dp-nyu-depth-v2-tf/README.md)| 0.573 | 63.5421 | 34.5255 |

## Image Inpainting

Image inpainting task is to estimate suitable pixel information to fill holes in images.

| Model Name                | Implementation | OMZ Model Name | Accuracy | GFlops | mParams |
| ------------------------- | ---------------| -------------- | -------- | ------ | ------- |
| GMCNN Inpainting          | TensorFlow\*   | [gmcnn-places2-tf](./gmcnn-places2-tf/README.md) | 33.47Db | 691.1589 | 12.7773|

## Style Transfer

Style transfer task is to transfer the style of one image to another.

| Model Name                     | Implementation | OMZ Model Name | Accuracy | GFlops | mParams |
| ------------------------------ | ---------------| -------------- | -------- | ------ | ------- |
| fast-neural-style-mosaic-onnx  | ONNX\*         | [fast-neural-style-mosaic-onnx](./fast-neural-style-mosaic-onnx/README.md) | 12.04dB | 15.518 | 1.679 |

## Action Recognition

The task of action recognition is to predict action that is being performed on a short video clip
(tensor formed by stacking sampled frames from input video).

| Model Name                        | Implementation | OMZ Model Name | Accuracy | GFlops | mParams |
| --------------------------------- | ---------------| -------------- | -------- | ------ | ------- |
| RGB-I3D, pretrained on ImageNet\* | TensorFlow\*   | [i3d-rgb-tf](./i3d-rgb-tf/README.md) | 65.96%/86.01% | 278.9815 | 12.6900|
| common-sign-language-0001         | PyTorch\*      | [common-sign-language-0001](./common-sign-language-0001/README.md) | 93.58% | 4.2269 | 4.1128 |

## Colorization

Colorization task is to predict colors of scene from grayscale image.

| Model Name                | Implementation | OMZ Model Name | Accuracy | GFlops | mParams |
| ------------------------- | ---------------| -------------- | -------- | ------ | ------- |
| colorization-v2           | PyTorch\* | [colorization-v2](./colorization-v2/README.md) | 26.99dB | 83.6045 |  32.2360 |
| colorization-siggraph     | PyTorch\* | [colorization-siggraph](./colorization-siggraph/README.md) | 27.73dB | 150.5441 |  34.0511 |

## Sound Classification

The task of sound classification is to predict what sounds are in an audio fragment.

| Model Name                | Implementation | OMZ Model Name | Accuracy | GFlops | mParams |
| ------------------------- | ---------------| -------------- | ------ | ------- | ------- |
| ACLNet                    | PyTorch\* | [aclnet](./aclnet/README.md) | 86%/92% | 1.4     | 2.7     |
| ACLNet-int8               | PyTorch\* | [aclnet-int8](./aclnet-int8/README.md) | 87%/93%   | 1.41     | 2.71     |

## Speech Recognition

The task of speech recognition is to recognize and translate spoken language into text.

| Model Name        | Implementation | OMZ Model Name                                               | Accuracy | GFlops | mParams |
| ----------------- | -------------- | ------------------------------------------------------------ | -------- | ------ | ------- |
| DeepSpeech V0.6.1 | TensorFlow\*   | [mozilla-deepspeech-0.6.1](./mozilla-deepspeech-0.6.1/README.md) | 7.55%    | 0.0472 | 47.2    |
| DeepSpeech V0.8.2 | TensorFlow\*   | [mozilla-deepspeech-0.8.2](./mozilla-deepspeech-0.8.2/README.md) | 6.13%    | 0.0472 | 47.2    |
| QuartzNet | Pytorch\* | [quartznet-15x5-en](./quartznet-15x5-en/README.md) | 3.86% | 2.4195 | 18.8857 |

## Image Translation

The task of image translation is to generate the output based on exemplar.

| Model Name | Implementation | OMZ Model Name                     | Accuracy | GFlops    | mParams  |
| -----------| -------------- | ---------------------------------- | -------- | --------- | -------- |
| CoCosNet   | PyTorch\*      | [cocosnet](./cocosnet/README.md) | 12.93dB  | 1080.7032 | 167.9141 |

## Optical Character Recognition

| Model Name | Implementation | OMZ Model Name                     | Accuracy | GFlops    | mParams  |
| -----------| -------------- | ---------------------------------- | -------- | --------- | -------- |
| license-plate-recognition-barrier-0007 | TensorFlow\* | [license-plate-recognition-barrier-0007](./license-plate-recognition-barrier-0007/README.md) | 98% | 0.347 | 1.435 |

## Place Recognition

The task of place recognition is to quickly and accurately recognize the location of a given query photograph.

| Model Name | Implementation | OMZ Model Name                           | Accuracy | GFlops | mParams |
| ---------- | ---------------| -----------------------------------------| -------- | ------ | ------- |
| NetVLAD    | TensorFlow\*   | [netvlad-tf](./netvlad-tf/README.md) | 82.0321% | 36.6374| 149.0021|

## Deblurring

The task of image deblurring.

| Model Name     | Implementation | OMZ Model Name                                 | Accuracy | GFlops  | mParams  |
| -------------- | -------------- | ---------------------------------------------- | -------- | ------- | -------- |
| DeblurGAN-v2   | PyTorch\*      | [deblurgan-v2](./deblurgan-v2/README.md) | 28.25Db  | 80.8919 | 2.1083   |

## Salient object detection

 Salient object detection is a task-based on a visual attention mechanism,
 in which algorithms aim to explore objects or regions more attentive than the surrounding areas on the scene or images.

| Model Name     | Implementation | OMZ Model Name            | Accuracy | GFlops  | mParams  |
| -------------- | -------------- | ------------------------- | -------- | ------- | -------- |
| F3Net          | PyTorch\*      | [f3net](./f3net/README.md) | 84.21%   | 31.2883 | 25.2791  |

## Text Recognition

Scene text recognition is a task to recognize text on a given image.
Researchers compete on creating algorithms which are able to recognize text of different shapes, fonts and background.
See details about datasets in [here](./text-recognition-resnet-fc/README.md)
The reported metric is collected over the alphanumeric subset of icdar 13 (1015 images) in case-insensitive mode.

| Model Name     | Implementation | OMZ Model Name                                         | Accuracy | GFlops  | mParams  |
| -------------- | -------------- | ------------------------------------------------------ | -------- | ------- | -------- |
| Resnet-FC      | PyTorch\*      | [text-recognition-resnet-fc](./text-recognition-resnet-fc/README.md) | 90.94% | 40.3704 | 177.9668  |

## Text to Speech

| Model Name     | Implementation | OMZ Model Name                                         | Accuracy | GFlops  | mParams  |
| -------------- | -------------- | ------------------------------------------------------ | -------- | ------- | -------- |
| ForwardTacotron | PyTorch\* | [forward-tacotron](./forward-tacotron/README.md):<br>forward-tacotron-duration-prediction <br>forward-tacotron-regression |  | <br>6.66 <br>4.91 | <br>13.81 <br>3.05 |
| WaveRNN        | PyTorch\*      | [wavernn](./wavernn/README.md):<br>wavernn-upsampler <br>wavernn-rnn |  | <br>0.37 <br>0.06 | <br>0.4 <br>3.83 |

## Named Entity Recognition

Named entity recognition (NER) is the task of tagging entities in text with their corresponding type.

| Model Name     | Implementation | OMZ Model Name                                   | Accuracy | GFlops  | mParams  |
| -------------- | -------------- | ------------------------------------------------ | -------- | ------- | --------- |
| bert-base-NER  | PyTorch\*      | [bert-base-ner](./bert-base-ner/README.md) | 94.45%  | 22.3874 | 107.4319  |

## Vehicle Reidentification

| Model Name     | Implementation | OMZ Model Name                                         | Accuracy | GFlops  | mParams  |
| -------------- | -------------- | ------------------------------------------------------ | -------- | ------- | -------- |
| vehicle-reid-0001 | PyTorch\* | [vehicle-reid-0001](./vehicle-reid-0001/README.md) | 96.31%/85.15 % | 2.643 | 2.183 |

## See Also

* [Open Model Zoo Demos](../../demos/README.md)
* [Model Downloader](../../tools/downloader/README.md)
* [Overview of OpenVINO&trade; Toolkit Intel's Pre-Trained Models](../intel/index.md)

## Legal Information

[*] Other names and brands may be claimed as the property of others.
