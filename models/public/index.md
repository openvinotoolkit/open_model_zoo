# Overview of OpenVINO&trade; Toolkit Public Pre-Trained Models

<!--
@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Device Support

   omz_models_public_device_support
   omz_models_model_aclnet
   omz_models_model_aclnet_int8
   omz_models_model_anti_spoof_mn3
   omz_models_model_background_matting_mobilenetv2
   omz_models_model_bert_base_ner
   omz_models_model_brain_tumor_segmentation_0002
   omz_models_model_cocosnet
   omz_models_model_colorization_siggraph
   omz_models_model_colorization_v2
   omz_models_model_common_sign_language_0001
   omz_models_model_convnext_tiny
   omz_models_model_ctdet_coco_dlav0_512
   omz_models_model_ctpn
   omz_models_model_deeplabv3
   omz_models_model_densenet_121_tf
   omz_models_model_detr_resnet50
   omz_models_model_dla_34
   omz_models_model_drn_d_38
   omz_models_model_efficientdet_d0_tf
   omz_models_model_efficientdet_d1_tf
   omz_models_model_efficientnet_b0
   omz_models_model_efficientnet_b0_pytorch
   omz_models_model_efficientnet_v2_b0
   omz_models_model_efficientnet_v2_s
   omz_models_model_erfnet
   omz_models_model_f3net
   omz_models_model_face_recognition_resnet100_arcface_onnx
   omz_models_model_faceboxes_pytorch
   omz_models_model_facenet_20180408_102900
   omz_models_model_fast_neural_style_mosaic_onnx
   omz_models_model_faster_rcnn_inception_resnet_v2_atrous_coco
   omz_models_model_faster_rcnn_resnet50_coco
   omz_models_model_fastseg_large
   omz_models_model_fastseg_small
   omz_models_model_fbcnn
   omz_models_model_fcrn_dp_nyu_depth_v2_tf
   omz_models_model_forward_tacotron
   omz_models_model_gmcnn_places2_tf
   omz_models_model_googlenet_v1_tf
   omz_models_model_googlenet_v2_tf
   omz_models_model_googlenet_v3
   omz_models_model_googlenet_v3_pytorch
   omz_models_model_googlenet_v4_tf
   omz_models_model_gpt_2
   omz_models_model_hbonet_0_25
   omz_models_model_hbonet_1_0
   omz_models_model_higher_hrnet_w32_human_pose_estimation
   omz_models_model_hrnet_v2_c1_segmentation
   omz_models_model_human_pose_estimation_3d_0001
   omz_models_model_hybrid_cs_model_mri
   omz_models_model_i3d_rgb_tf
   omz_models_model_inception_resnet_v2_tf
   omz_models_model_levit_128s
   omz_models_model_license_plate_recognition_barrier_0007
   omz_models_model_mask_rcnn_inception_resnet_v2_atrous_coco
   omz_models_model_mask_rcnn_resnet50_atrous_coco
   omz_models_model_midasnet
   omz_models_model_mixnet_l
   omz_models_model_mobilenet_v1_0_25_128
   omz_models_model_mobilenet_v1_1_0_224_tf
   omz_models_model_mobilenet_v2_1_0_224
   omz_models_model_mobilenet_v2_1_4_224
   omz_models_model_mobilenet_v2_pytorch
   omz_models_model_mobilenet_v3_large_1_0_224_tf
   omz_models_model_mobilenet_v3_small_1_0_224_tf
   omz_models_model_mobilenet_yolo_v4_syg
   omz_models_model_modnet_photographic_portrait_matting
   omz_models_model_modnet_webcam_portrait_matting
   omz_models_model_mozilla_deepspeech_0_6_1
   omz_models_model_mozilla_deepspeech_0_8_2
   omz_models_model_nanodet_m_1_5x_416
   omz_models_model_nanodet_plus_m_1_5x_416
   omz_models_model_netvlad_tf
   omz_models_model_nfnet_f0
   omz_models_model_open_closed_eye_0001
   omz_models_model_pspnet_pytorch
   omz_models_model_quartznet_15x5_en
   omz_models_model_regnetx_3_2gf
   omz_models_model_repvgg_a0
   omz_models_model_repvgg_b1
   omz_models_model_repvgg_b3
   omz_models_model_resnest_50_pytorch
   omz_models_model_resnet_18_pytorch
   omz_models_model_resnet_34_pytorch
   omz_models_model_resnet_50_pytorch
   omz_models_model_resnet_50_tf
   omz_models_model_retinaface_resnet50_pytorch
   omz_models_model_retinanet_tf
   omz_models_model_rexnet_v1_x1_0
   omz_models_model_rfcn_resnet101_coco_tf
   omz_models_model_robust_video_matting_mobilenetv3
   omz_models_model_shufflenet_v2_x1_0
   omz_models_model_single_human_pose_estimation_0001
   omz_models_model_ssd_mobilenet_v1_coco
   omz_models_model_ssd_mobilenet_v1_fpn_coco
   omz_models_model_ssd_resnet34_1200_onnx
   omz_models_model_ssdlite_mobilenet_v2
   omz_models_model_swin_tiny_patch4_window7_224
   omz_models_model_t2t_vit_14
   omz_models_model_text_recognition_resnet_fc
   omz_models_model_ultra_lightweight_face_detection_rfb_320
   omz_models_model_ultra_lightweight_face_detection_slim_320
   omz_models_model_vehicle_license_plate_detection_barrier_0123
   omz_models_model_vehicle_reid_0001
   omz_models_model_vitstr_small_patch16_224
   omz_models_model_wav2vec2_base
   omz_models_model_wavernn
   omz_models_model_yolact_resnet50_fpn_pytorch
   omz_models_model_yolo_v1_tiny_tf
   omz_models_model_yolo_v2_tf
   omz_models_model_yolo_v2_tiny_tf
   omz_models_model_yolo_v3_onnx
   omz_models_model_yolo_v3_tf
   omz_models_model_yolo_v3_tiny_onnx
   omz_models_model_yolo_v3_tiny_tf
   omz_models_model_yolo_v4_tf
   omz_models_model_yolo_v4_tiny_tf
   omz_models_model_yolof
   omz_models_model_yolox_tiny


@endsphinxdirective
-->

OpenVINO&trade; toolkit provides a set of public pre-trained models
that you can use for learning and demo purposes or for developing deep learning
software. Most recent version is available in the [repo on Github](https://github.com/openvinotoolkit/open_model_zoo).
The table [Public Pre-Trained Models Device Support](./device_support.md) summarizes devices supported by each model.

You can download models and convert them into OpenVINO™ IR format (\*.xml + \*.bin) using the OpenVINO™ [Model Downloader](../../tools/model_tools/README.md) and other automation tools.

## Classification Models

<div class="sort-table"></div>

| Model Name                  | Implementation                     | OMZ Model Name | Accuracy | GFlops | mParams |
| --------------------------- | -----------------------------------| -------------- | -------- | ------ | ------- |
| AntiSpoofNet                | PyTorch\*                          | [anti-spoof-mn3](./anti-spoof-mn3/README.md) | 3.81% | 0.15 | 3.02 |
| ConvNeXt Tiny               | PyTorch\*                          | [convnext-tiny](./convnext-tiny/README.md) | 82.05%/95.86% | 8.9419 | 28.5892 |
| DenseNet 121                | [densenet-121-tf](./densenet-121-tf/README.md)| 74.46%/92.13%| 5.723~5.7287 | 7.971 |
| DLA 34                      | PyTorch\*                          | [dla-34](./dla-34/README.md) | 74.64%/92.06% | 6.1368 | 15.7344 |
| EfficientNet B0             | TensorFlow\*<br>PyTorch\*          | [efficientnet-b0](./efficientnet-b0/README.md)<br>[efficientnet-b0-pytorch](./efficientnet-b0-pytorch/README.md) | 75.70%/92.76%<br>77.70%/93.52% | 0.819 | 5.268 |
| EfficientNet V2 B0          | PyTorch\*                          | [efficientnet-v2-b0](./efficientnet-v2-b0/README.md) | 78.36%/94.02% | 1.4641 | 7.1094 |
| EfficientNet V2 Small       | PyTorch\*                          | [efficientnet-v2-s](./efficientnet-v2-s/README.md) | 84.29%/97.26% | 16.9406  | 21.3816  |
| HBONet 1.0                  | PyTorch\*                          | [hbonet-1.0](./hbonet-1.0/README.md)  | 73.1%/91.0% | 0.6208 | 4.5443 |
| HBONet 0.25                 | PyTorch\*                          | [hbonet-0.25](./hbonet-0.25/README.md) | 57.3%/79.8% | 0.0758 | 1.9299 |
| Inception (GoogleNet) V1    | TensorFlow\*            | [googlenet-v1-tf](./googlenet-v1-tf/README.md) | 69.814%/89.6% | 3.016~3.266 | 6.619~6.999 |
| Inception (GoogleNet) V2    | TensorFlow\*            | [googlenet-v2-tf](./googlenet-v2-tf/README.md) | 74.084%/91.798%| 4.058 | 11.185 |
| Inception (GoogleNet) V3    | TensorFlow\*<br>PyTorch\*          | [googlenet-v3](./googlenet-v3/README.md) <br> [googlenet-v3-pytorch](./googlenet-v3-pytorch/README.md) | 77.904%/93.808%<br>77.69%/93.7% | 11.469 | 23.817 |
| Inception (GoogleNet) V4    | TensorFlow\*                       | [googlenet-v4-tf](./googlenet-v4-tf/README.md) | 80.204%/95.21% | 24.584 | 42.648 |
| Inception-ResNet V2         | TensorFlow\*                       | [inception-resnet-v2-tf](./inception-resnet-v2-tf/README.md) | 77.82%/94.03% | 22.227 | 30.223 |
| LeViT 128S                  | PyTorch\*                          | [levit-128s](./levit-128s/README.md) | 76.54%/92.85% | 0.6177 | 8.2199 |
| MixNet L                    | TensorFlow\*                       | [mixnet-l](./mixnet-l/README.md)  | 78.30%/93.91% | 0.565 | 7.3 |
| MobileNet V1 0.25 128       | Caffe\*                            | [mobilenet-v1-0.25-128](./mobilenet-v1-0.25-128/README.md)  | 40.54%/65% | 0.028 | 0.468 |
| MobileNet V1 1.0 224        | Caffe\*<br>TensorFlow\*            | [mobilenet-v1-1.0-224-tf](./mobilenet-v1-1.0-224-tf/README.md)| 71.03%/89.94% | 1.148 | 4.221 |
| MobileNet V2 1.0 224        | TensorFlow\*<br>PyTorch\*| [mobilenet-v2-1.0-224](./mobilenet-v2-1.0-224/README.md)<br>[mobilenet-v2-pytorch](./mobilenet-v2-pytorch/README.md) | 71.85%/90.69%<br>71.81%/90.396% | 0.615~0.876 | 3.489 |
| MobileNet V2 1.4 224        | TensorFlow\*                       | [mobilenet-v2-1.4-224](./mobilenet-v2-1.4-224/README.md) | 74.09%/91.97% | 1.183  | 6.087 |
| MobileNet V3 Small 1.0      | TensorFlow\* | [mobilenet-v3-small-1.0-224-tf](./mobilenet-v3-small-1.0-224-tf/README.md) | 67.36%/87.44% | 0.1168 | 2.537 |
| MobileNet V3 Large 1.0      | TensorFlow\*                       | [mobilenet-v3-large-1.0-224-tf](./mobilenet-v3-large-1.0-224-tf/README.md) | 75.30%/92.62% | 0.4450 | 5.4721 |
| NFNet F0                    | PyTorch\*                          | [nfnet-f0](./nfnet-f0/README.md) | 83.34%/96.56% | 24.8053 | 71.4444 |
| RegNetX-3.2GF               | PyTorch\*                          | [regnetx-3.2gf](./regnetx-3.2gf/README.md) | 78.17%/94.08% | 6.3893 | 15.2653 |
| open-closed-eye-0001        | PyTorch\*                          | [open-closed-eye-0001](./open-closed-eye-0001/README.md) |  95.84%   | 0.0014 | 0.0113|
| RepVGG A0                   | PyTorch\*                          | [repvgg-a0](./repvgg-a0/README.md) | 72.40%/90.49% | 2.7286 | 8.3094 |
| RepVGG B1                   | PyTorch\*                          | [repvgg-b1](./repvgg-b1/README.md) | 78.37%/94.09% | 23.6472 | 51.8295|
| RepVGG B3                   | PyTorch\*                          | [repvgg-b3](./repvgg-b3/README.md) | 80.50%/95.25% | 52.4407 | 110.9609|
| ResNeSt 50                  | PyTorch\*                          | [resnest-50-pytorch](./resnest-50-pytorch/README.md) | 81.11%/95.36% | 10.8148 |  27.4493|
| ResNet 18                   | PyTorch\*                          | [resnet-18-pytorch](./resnet-18-pytorch/README.md) | 69.754%/89.088% | 3.637 |  11.68 |
| ResNet 34                   | PyTorch\*                          | [resnet-34-pytorch](./resnet-34-pytorch/README.md) | 73.30%/91.42% | 7.3409 | 21.7892 |
| ResNet 50                   | PyTorch\*<br>TensorFlow\*| [resnet-50-pytorch](./resnet-50-pytorch/README.md)[resnet-50-tf](./resnet-50-tf/README.md)| 75.168%/92.212%<br>76.38%/93.188%<br>76.17%/92.98% | 6.996~8.216 | 25.53 |
| ReXNet V1 x1.0              | PyTorch\*                          | [rexnet-v1-x1.0](./rexnet-v1-x1.0/README.md) | 77.86%/93.87% | 0.8325 | 4.7779 |
| Shufflenet V2 x1.0          | PyTorch\*                          | [shufflenet-v2-x1.0](./shufflenet-v2-x1.0/README.md) | 69.36%/88.32% | 0.2957 | 2.2705 |
| Swin Transformer Tiny, window size=7| PyTorch\*                  | [swin-tiny-patch4-window7-224](./swin-tiny-patch4-window7-224/README.md) | 81.38%/95.51% | 9.0280 | 28.8173 |
| T2T-ViT, transformer layers number=14| PyTorch\*                 | [t2t-vit-14](./t2t-vit-14/README.md) | 81.44%/95.66% | 9.5451 | 21.5498 |

## Segmentation Models

Semantic segmentation is an extension of object detection problem. Instead of
returning bounding boxes, semantic segmentation models return a "painted"
version of the input image, where the "color" of each pixel represents a certain
class. These networks are much bigger than respective object detection networks,
but they provide a better (pixel-level) localization of objects and they can
detect areas with complex shape.

### Semantic Segmentation Models

<div class="sort-table"></div>

| Model Name                | Implementation | OMZ Model Name | Accuracy | GFlops | mParams |
| ------------------------- | -------------- | -------------- | -------- | ------ | ------- |
| DeepLab V3                | TensorFlow\*   | [deeplabv3](./deeplabv3/README.md) | 68.41% | 11.469 | 23.819 |
| DRN-D-38                  |  PyTorch\*     | [drn-d-38](./drn-d-38/README.md) | 71.31% | 1768.3276 | 25.9939 |
| Erfnet | PyTorch\*      | [erfnet](./erfnet/README.md)| 76.47% | 11.13 | 7.87 |
| HRNet V2 C1 Segmentation  | PyTorch\*      | [hrnet-v2-c1-segmentation](./hrnet-v2-c1-segmentation/README.md) | 77.69% | 81.993 | 66.4768 |
| Fastseg MobileV3Large LR-ASPP, F=128  | PyTorch\*      | [fastseg-large](./fastseg-large/README.md) | 72.67% | 140.9611 | 3.2 |
| Fastseg MobileV3Small LR-ASPP, F=128  | PyTorch\*      | [fastseg-small](./fastseg-small/README.md) | 67.15% | 69.2204 | 1.1 |
| PSPNet R-50-D8  | PyTorch\*      | [pspnet-pytorch](./pspnet-pytorch/README.md) | 70.6% | 357.1719 | 46.5827 |

### Instance Segmentation Models

Instance segmentation is an extension of object detection and semantic
segmentation problems. Instead of predicting a bounding box around each object
instance instance segmentation model outputs pixel-wise masks for all instances.

<div class="sort-table"></div>

| Model Name                     | Implementation | OMZ Model Name | Accuracy | GFlops | mParams |
| ------------------------------ | -------------- | -------------- | -------- | ------ | ------- |
| Mask R-CNN Inception ResNet V2 | TensorFlow\*   | [mask_rcnn_inception_resnet_v2_atrous_coco](./mask_rcnn_inception_resnet_v2_atrous_coco/README.md) | 39.86%/35.36% | 675.314 | 92.368 |
| Mask R-CNN ResNet 50           | TensorFlow\*   | [mask_rcnn_resnet50_atrous_coco](./mask_rcnn_resnet50_atrous_coco/README.md)| 	29.75%/27.46% | 294.738 | 50.222 |
| YOLACT ResNet 50 FPN | PyTorch\* | [yolact-resnet50-fpn-pytorch](./yolact-resnet50-fpn-pytorch/README.md) | 28.0%/30.69% | 118.575 |  36.829  |

### 3D Semantic Segmentation Models

<div class="sort-table"></div>

| Model Name                | Implementation | OMZ Model Name | Accuracy | GFlops | mParams |
| ------------------------- | -------------- | -------------- | -------- | ------ | ------- |
| Brain Tumor Segmentation 2| PyTorch\*      | [brain-tumor-segmentation-0002](./brain-tumor-segmentation-0002/README.md) | 91.4826% | 300.801 | 4.51  |

## Object Detection Models

Several detection models can be used to detect a set of the most popular
objects - for example, faces, people, vehicles. Most of the networks are
SSD-based and provide reasonable accuracy/performance trade-offs.

<div class="sort-table"></div>

| Model Name                           | Implementation           | OMZ Model Name | Accuracy | GFlops | mParams |
| ------------------------------------ | ------------------------ | -------------- | -------- | ------ | ------- |
| CTPN                                 | TensorFlow\*             | [ctpn](./ctpn/README.md) | 73.67% | 55.813 | 17.237 |
| CenterNet (CTDET with DLAV0) 512x512 | ONNX\*                   | [ctdet_coco_dlav0_512](./ctdet_coco_dlav0_512/README.md)| 44.2756%| 62.211 | 17.911 |
| DETR-ResNet50                        | PyTorch\*                | [detr-resnet50](./detr-resnet50/README.md)| 39.27% / 42.36% | 174.4708 | 41.3293 |
| EfficientDet-D0                      | TensorFlow\*             | [efficientdet-d0-tf](./efficientdet-d0-tf/README.md)| 31.95% | 2.54 | 3.9 |
| EfficientDet-D1                      | TensorFlow\*             | [efficientdet-d1-tf](./efficientdet-d1-tf/README.md)| 37.54% | 6.1 | 6.6 |
| FaceBoxes                            | PyTorch\*                | [faceboxes-pytorch](./faceboxes-pytorch/README.md)|83.565% | 1.8975 | 1.0059 |
| Faster R-CNN with Inception-ResNet v2| TensorFlow\*             | [faster_rcnn_inception_resnet_v2_atrous_coco](./faster_rcnn_inception_resnet_v2_atrous_coco/README.md)| 40.69% | 30.687 | 13.307 |
| Faster R-CNN with ResNet 50          | TensorFlow\*             | [faster_rcnn_resnet50_coco](./faster_rcnn_resnet50_coco/README.md) | 31.09% | 57.203 | 29.162 |
| Mobilenet-yolo-v4-syg                | Keras\*                  | [mobilenet-yolo-v4-syg](./mobilenet-yolo-v4-syg/README.md)| 	86.35%| 65.981 | 61.922 |
| NanoDet with ShuffleNetV2 1.5x, size=416 | PyTorch\*            | [nanodet-m-1.5x-416](./nanodet-m-1.5x-416/README.md) | 27.38%/26.63% | 2.3895 | 2.0534 |
| NanoDet Plus with ShuffleNetV2 1.5x, size=416 | PyTorch\*       | [nanodet-plus-m-1.5x-416](./nanodet-plus-m-1.5x-416/README.md) | 34.53%/33.77% | 3.0147 | 2.4614 |
| RetinaFace with ResNet 50            | PyTorch\*                | [retinaface-resnet50-pytorch](./retinaface-resnet50-pytorch/README.md) | 91.78% | 88.8627 | 27.2646 |
| RetinaNet with Resnet 50             | TensorFlow\*             | [retinanet-tf](./retinanet-tf/README.md) | 33.15% | 238.9469 | 64.9706 |
| R-FCN with Resnet-101                | TensorFlow\*             | [rfcn-resnet101-coco-tf](./rfcn-resnet101-coco-tf/README.md) | 28.40%/45.02% | 53.462 | 171.85 |
| SSD with MobileNet                   | TensorFlow\* | [ssd_mobilenet_v1_coco](./ssd_mobilenet_v1_coco/README.md) | 23.32%| 2.316~2.494 | 5.783~6.807 |
| SSD with MobileNet FPN               | TensorFlow\*             | [ssd_mobilenet_v1_fpn_coco](./ssd_mobilenet_v1_fpn_coco/README.md) | 35.5453% | 123.309 | 36.188 |
| SSD lite with MobileNet V2           | TensorFlow\*             | [ssdlite_mobilenet_v2](./ssdlite_mobilenet_v2/README.md) | 24.2946% | 1.525 | 4.475 |
| SSD with ResNet 34 1200x1200         | PyTorch\*                | [ssd-resnet34-1200-onnx](./ssd-resnet34-1200-onnx/README.md) | 20.7198%/39.2752% | 433.411 | 20.058  |
| Ultra Lightweight Face Detection RFB 320| PyTorch\*             | [ultra-lightweight-face-detection-rfb-320](./ultra-lightweight-face-detection-rfb-320/README.md)|84.78% | 0.2106 | 0.3004 |
| Ultra Lightweight Face Detection slim 320| PyTorch\*            | [ultra-lightweight-face-detection-slim-320](./ultra-lightweight-face-detection-slim-320/README.md)|83.32% | 0.1724 | 0.2844 |
| Vehicle License Plate Detection Barrier | TensorFlow\* | [vehicle-license-plate-detection-barrier-0123](./vehicle-license-plate-detection-barrier-0123/README.md) | 	99.52% | 0.271 | 0.547 |
| YOLO v1 Tiny                         | TensorFlow.js\*          | [yolo-v1-tiny-tf](./yolo-v1-tiny-tf/README.md) | 54.79% | 6.9883	 |	15.8587 |
| YOLO v2 Tiny                         | Keras\*                  | [yolo-v2-tiny-tf](./yolo-v2-tiny-tf/README.md) | 27.3443%/29.1184%| 5.4236	 |	11.2295 |
| YOLO v2                              | Keras\*                  | [yolo-v2-tf](./yolo-v2-tf/README.md) | 53.1453%/56.483% | 63.0301	 |	50.9526 |
| YOLO v3                              | Keras\* <br>ONNX\*       | [yolo-v3-tf](./yolo-v3-tf/README.md) <br>[yolo-v3-onnx](./yolo-v3-onnx/README.md) | 62.2759%/67.7221% <br> 48.30%/47.07%| 65.9843~65.998 | 61.9221~61.930 |
| YOLO v3 Tiny                         | Keras\* <br>ONNX\*       | [yolo-v3-tiny-tf](./yolo-v3-tiny-tf/README.md) <br>[yolo-v3-tiny-onnx](./yolo-v3-tiny-onnx/README.md) | 35.9%/39.7% <br> 17.07%/13.64%| 5.582  | 8.848~8.8509 |
| YOLO v4                              | Keras\*                  | [yolo-v4-tf](./yolo-v4-tf/README.md) | 71.23%/77.40%/50.26% | 129.5567	 |	64.33 |
| YOLO v4 Tiny                         | Keras\*                  | [yolo-v4-tiny-tf](./yolo-v4-tiny-tf/README.md) | | 6.9289 | 6.0535 |
| YOLOF                                | PyTorch\*                | [yolof](./yolof/README.md)           | 60.69%/66.23%/43.63% | 175.37942 | 48.228 |
| YOLOX Tiny                           | PyTorch\*                | [yolox-tiny](./yolox-tiny/README.md) | 47.85%/52.56%/31.82%| 6.4813 | 5.0472 |

## Face Recognition Models

<div class="sort-table"></div>

| Model Name                           | Implementation | OMZ Model Name | Accuracy | GFlops | mParams |
| ------------------------------------ | -------------- | -------------- | -------- |------ | ------- |
| FaceNet                              | TensorFlow\*   | [facenet-20180408-102900](./facenet-20180408-102900/README.md) | 99.14% | 2.846 | 23.469 |
| LResNet100E-IR,ArcFace@ms1m-refine-v2| MXNet\*        | [face-recognition-resnet100-arcface-onnx](./face-recognition-resnet100-arcface-onnx/README.md) | 99.68%| 24.2115  | 65.1320 |

## Human Pose Estimation Models

Human pose estimation task is to predict a pose: body skeleton, which consists
of keypoints and connections between them, for every person in an input image or
video. Keypoints are body joints, i.e. ears, eyes, nose, shoulders, knees, etc.
There are two major groups of such methods: top-down and bottom-up.  The first
detects persons in a given frame, crops or rescales detections, then runs pose
estimation network for every detection. These methods are very accurate. The
second finds all keypoints in a given frame, then groups them by person
instances, thus faster than previous, because network runs once.

<div class="sort-table"></div>

| Model Name                       | Implementation | OMZ Model Name                | Accuracy | GFlops | mParams |
|--------------------------------- | ---------------| ----------------------------- | -------- | ------ | ------- |
| human-pose-estimation-3d-0001    | PyTorch\*      | [human-pose-estimation-3d-0001](./human-pose-estimation-3d-0001/README.md) | 100.44437mm | 18.998 |  5.074  |
| single-human-pose-estimation-0001| PyTorch\*      | [single-human-pose-estimation-0001](./single-human-pose-estimation-0001/README.md) | 69.0491% | 60.125 | 33.165 |
| higher-hrnet-w32-human-pose-estimation | PyTorch\* | [higher-hrnet-w32-human-pose-estimation](./higher-hrnet-w32-human-pose-estimation/README.md) | 64.64%    | 92.8364 | 28.6180 |

## Monocular Depth Estimation Models

The task of monocular depth estimation is to predict a depth (or inverse depth) map based on a single input image.
Since this task contains - in the general setting - some ambiguity, the resulting depth maps are often only defined up to an unknown scaling factor.

<div class="sort-table"></div>

| Model Name                  | Implementation | OMZ Model Name                | Accuracy | GFlops    | mParams |
| --------------------------- | -------------- | ----------------------------- | -------- | --------- | ------- |
| midasnet                    | PyTorch\*      | [midasnet](./midasnet/README.md)| 0.07071 | 207.25144  |    104.081     |
| FCRN ResNet50-Upproj        | TensorFlow\*   | [fcrn-dp-nyu-depth-v2-tf](./fcrn-dp-nyu-depth-v2-tf/README.md)| 0.573 | 63.5421 | 34.5255 |

## Image Inpainting Models

Image inpainting task is to estimate suitable pixel information to fill holes in images.

<div class="sort-table"></div>

| Model Name                | Implementation | OMZ Model Name | Accuracy | GFlops | mParams |
| ------------------------- | ---------------| -------------- | -------- | ------ | ------- |
| GMCNN Inpainting          | TensorFlow\*   | [gmcnn-places2-tf](./gmcnn-places2-tf/README.md) | 33.47Db | 691.1589 | 12.7773|
| Hybrid-CS-Model-MRI       | TensorFlow\*   | [hybrid-cs-model-mri](./hybrid-cs-model-mri/README.md) | 34.27Db | 146.6037 | 11.3313 |

## Style Transfer Models

Style transfer task is to transfer the style of one image to another.

<div class="sort-table"></div>

| Model Name                     | Implementation | OMZ Model Name | Accuracy | GFlops | mParams |
| ------------------------------ | ---------------| -------------- | -------- | ------ | ------- |
| fast-neural-style-mosaic-onnx  | ONNX\*         | [fast-neural-style-mosaic-onnx](./fast-neural-style-mosaic-onnx/README.md) | 12.04dB | 15.518 | 1.679 |

## Action Recognition Models

The task of action recognition is to predict action that is being performed on a short video clip
(tensor formed by stacking sampled frames from input video).

<div class="sort-table"></div>

| Model Name                        | Implementation | OMZ Model Name | Accuracy | GFlops | mParams |
| --------------------------------- | ---------------| -------------- | -------- | ------ | ------- |
| RGB-I3D, pretrained on ImageNet\* | TensorFlow\*   | [i3d-rgb-tf](./i3d-rgb-tf/README.md) | 64.83%/84.58% | 278.9815 | 12.6900|
| common-sign-language-0001         | PyTorch\*      | [common-sign-language-0001](./common-sign-language-0001/README.md) | 93.58% | 4.2269 | 4.1128 |

## Colorization Models

Colorization task is to predict colors of scene from grayscale image.

<div class="sort-table"></div>

| Model Name                | Implementation | OMZ Model Name | Accuracy | GFlops | mParams |
| ------------------------- | ---------------| -------------- | -------- | ------ | ------- |
| colorization-v2           | PyTorch\* | [colorization-v2](./colorization-v2/README.md) | 26.99dB | 83.6045 |  32.2360 |
| colorization-siggraph     | PyTorch\* | [colorization-siggraph](./colorization-siggraph/README.md) | 27.73dB | 150.5441 |  34.0511 |

## Sound Classification Models

The task of sound classification is to predict what sounds are in an audio fragment.

<div class="sort-table"></div>

| Model Name                | Implementation | OMZ Model Name | Accuracy | GFlops | mParams |
| ------------------------- | ---------------| -------------- | ------ | ------- | ------- |
| ACLNet                    | PyTorch\* | [aclnet](./aclnet/README.md) | 86%/92% | 1.4     | 2.7     |
| ACLNet-int8               | PyTorch\* | [aclnet-int8](./aclnet-int8/README.md) | 87%/93%   | 1.41     | 2.71     |

## Speech Recognition Models

The task of speech recognition is to recognize and translate spoken language into text.

<div class="sort-table"></div>

| Model Name        | Implementation | OMZ Model Name                                                   | Accuracy | GFlops | mParams |
| ----------------- | -------------- | ---------------------------------------------------------------- | -------- | ------ | ------- |
| DeepSpeech V0.6.1 | TensorFlow\*   | [mozilla-deepspeech-0.6.1](./mozilla-deepspeech-0.6.1/README.md) | 7.55%    | 0.0472 | 47.2    |
| DeepSpeech V0.8.2 | TensorFlow\*   | [mozilla-deepspeech-0.8.2](./mozilla-deepspeech-0.8.2/README.md) | 6.13%    | 0.0472 | 47.2    |
| QuartzNet         | PyTorch\*      | [quartznet-15x5-en](./quartznet-15x5-en/README.md)               | 3.86%    | 2.4195 | 18.8857 |
| Wav2Vec 2.0 Base  | PyTorch\*      | [wav2vec2-base](./wav2vec2-base/README.md)                       | 3.39%    | 26.843 | 94.3965 |

## Image Translation Models

The task of image translation is to generate the output based on exemplar.

<div class="sort-table"></div>

| Model Name | Implementation | OMZ Model Name                     | Accuracy | GFlops    | mParams  |
| -----------| -------------- | ---------------------------------- | -------- | --------- | -------- |
| CoCosNet   | PyTorch\*      | [cocosnet](./cocosnet/README.md) | 12.93dB  | 1080.7032 | 167.9141 |

## Optical Character Recognition Models

<div class="sort-table"></div>

| Model Name | Implementation | OMZ Model Name                     | Accuracy | GFlops    | mParams  |
| -----------| -------------- | ---------------------------------- | -------- | --------- | -------- |
| license-plate-recognition-barrier-0007 | TensorFlow\* | [license-plate-recognition-barrier-0007](./license-plate-recognition-barrier-0007/README.md) | 98% | 0.347 | 1.435 |

## Place Recognition Models

The task of place recognition is to quickly and accurately recognize the location of a given query photograph.

<div class="sort-table"></div>

| Model Name | Implementation | OMZ Model Name                           | Accuracy | GFlops | mParams |
| ---------- | ---------------| -----------------------------------------| -------- | ------ | ------- |
| NetVLAD    | TensorFlow\*   | [netvlad-tf](./netvlad-tf/README.md) | 82.0321% | 36.6374| 149.0021|

## JPEG Artifacts Removal Models

The task of restoration images from jpeg format.

<div class="sort-table"></div>

| Model Name     | Implementation | OMZ Model Name                                 | Accuracy | GFlops     | mParams  |
| -------------- | -------------- | ---------------------------------------------- | -------- | ---------- | -------- |
| FBCNN          | PyTorch\*      | [fbcnn](./fbcnn/README.md)                     | 34.34Db  | 1420.78235 | 71.922   |

## Salient Object Detection Models

 Salient object detection is a task-based on a visual attention mechanism,
 in which algorithms aim to explore objects or regions more attentive than the surrounding areas on the scene or images.

<div class="sort-table"></div>

| Model Name     | Implementation | OMZ Model Name            | Accuracy | GFlops  | mParams  |
| -------------- | -------------- | ------------------------- | -------- | ------- | -------- |
| F3Net          | PyTorch\*      | [f3net](./f3net/README.md) | 84.21%   | 31.2883 | 25.2791  |

## Text Prediction Models

Text prediction is a task to predict the next word, given all of the previous words within some text.

<div class="sort-table"></div>

| Model Name     | Implementation | OMZ Model Name             | Accuracy | GFlops   | mParams  |
| -------------- | -------------- | -------------------------- | -------- | -------- | -------- |
| GPT-2          | PyTorch\*      | [gpt-2](./gpt-2/README.md) | 29.00%   | 293.0489 | 175.6203 |

## Text Recognition Models

Scene text recognition is a task to recognize text on a given image.
Researchers compete on creating algorithms which are able to recognize text of different shapes, fonts and background.
See details about datasets in [here](./text-recognition-resnet-fc/README.md)
The reported metric is collected over the alphanumeric subset of ICDAR13 (1015 images) in case-insensitive mode.

<div class="sort-table"></div>

| Model Name                      | Implementation | OMZ Model Name                                                       | Accuracy | GFlops  | mParams  |
| ------------------------------- | -------------- | -------------------------------------------------------------------- | -------- | ------- | -------- |
| Resnet-FC                       | PyTorch\*      | [text-recognition-resnet-fc](./text-recognition-resnet-fc/README.md) | 90.94%   | 40.3704 | 177.9668 |
| ViTSTR Small patch=16, size=224 | PyTorch\*      | [vitstr-small-patch16-224](./vitstr-small-patch16-224/README.md)     | 90.34%   | 9.1544  | 21.5061  |

## Text to Speech Models

<div class="sort-table"></div>

| Model Name     | Implementation | OMZ Model Name                                         | Accuracy | GFlops  | mParams  |
| -------------- | -------------- | ------------------------------------------------------ | -------- | ------- | -------- |
| ForwardTacotron | PyTorch\* | [forward-tacotron](./forward-tacotron/README.md):<br>forward-tacotron-duration-prediction <br>forward-tacotron-regression |  | <br>6.66 <br>4.91 | <br>13.81 <br>3.05 |
| WaveRNN        | PyTorch\*      | [wavernn](./wavernn/README.md):<br>wavernn-upsampler <br>wavernn-rnn |  | <br>0.37 <br>0.06 | <br>0.4 <br>3.83 |

## Named Entity Recognition Models

Named entity recognition (NER) is the task of tagging entities in text with their corresponding type.

<div class="sort-table"></div>

| Model Name     | Implementation | OMZ Model Name                                   | Accuracy | GFlops  | mParams  |
| -------------- | -------------- | ------------------------------------------------ | -------- | ------- | --------- |
| bert-base-NER  | PyTorch\*      | [bert-base-ner](./bert-base-ner/README.md) | 94.45%  | 22.3874 | 107.4319  |

## Vehicle Reidentification Models

<div class="sort-table"></div>

| Model Name     | Implementation | OMZ Model Name                                         | Accuracy | GFlops  | mParams  |
| -------------- | -------------- | ------------------------------------------------------ | -------- | ------- | -------- |
| vehicle-reid-0001 | PyTorch\* | [vehicle-reid-0001](./vehicle-reid-0001/README.md) | 96.31%/85.15 % | 2.643 | 2.183 |

## Background Matting Models

Background matting is a method of separating a foreground from a background in an image or video,
wherein some pixels may belong to foreground as well as background, such pixels are called partial
or mixed pixels. This distinguishes background matting from segmentation approaches where the result is a binary mask.

<div class="sort-table"></div>

| Model Name     | Implementation | OMZ Model Name                                         | Accuracy | GFlops  | mParams  |
| -------------- | -------------- | ------------------------------------------------------ | -------- | ------- | -------- |
| background-matting-mobilenetv2 | PyTorch\* | [background-matting-mobilenetv2](./background-matting-mobilenetv2/README.md) | 4.32/1.0/2.48/2.7 | 6.7419 | 5.052 |
| modnet-photographic-portrait-matting | PyTorch\* | [modnet-photographic-portrait-matting](./modnet-photographic-portrait-matting/README.md) | 5.21/727.95 | 31.1564 | 6.4597 |
| modnet-webcam-portrait-matting | PyTorch\* | [modnet-webcam-portrait-matting](./modnet-webcam-portrait-matting/README.md) | 5.66/762.52 | 31.1564 | 6.4597 |
| robust-video-matting-mobilenetv3 | PyTorch\* | [robust-video-matting-mobilenetv3](./robust-video-matting-mobilenetv3/README.md) | 20.8/15.1/4.42/4.05 | 9.3892 | 3.7363 |

## See Also

* [Open Model Zoo Demos](../../demos/README.md)
* [Model Downloader](../../tools/model_tools/README.md)
* [Overview of OpenVINO&trade; Toolkit Intel's Pre-Trained Models](../intel/index.md)

## Legal Information

Caffe, Caffe2, Keras, MXNet, PyTorch, and TensorFlow are trademarks or brand names of their respective owners.
All company, product and service names used in this website are for identification purposes only.
Use of these names,trademarks and brands does not imply endorsement.
