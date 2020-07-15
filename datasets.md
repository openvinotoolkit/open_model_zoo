# Dataset Preparation Guide

If you want to use prepared configs to run the Accuracy Checker tool and the Model Quantizer, you need to organize `<DATASET_DIR>` folder with validation datasets in a certain way. Instructions for preparing validation data are described in this document.

## [ImageNet](http://image-net.org)

### How download dataset

To download images from ImageNet, you need to have an account and agree to the Terms of Access. Follow the steps below:
1. Go to the [ImageNet](http://www.image-net.org/) homepage
2. If you have an account, click `Login`. Otherwise, click `Signup` in the right upper corner, provide your data, and wait for a confirmation email
3. Log in after receiving the confirmation email and go to the `Download` tab
4. Select `Download Original Images`
5. You will be redirected to the Terms of Access page. If you agree to the Terms, continue by clicking Agree and Sign
6. Click one of the links in the `Download as one tar file` section to select it
7. Unpack archive

To download annotation files, you need to follow the steps below:
* `val.txt`
  1. Download [arhive](http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz)
  2. Unpack `val.txt` from the archive `caffe_ilsvrc12.tar.gz`
* `val15.txt`
  1. Download [annotation file](https://raw.githubusercontent.com/hujie-frank/SENet/master/ILSVRC2017_val.txt)
  2. Rename `ILSVRC2017_val.txt` to `val15.txt`

### Files layout

To use this dataset with OMZ tools, make sure `<DATASET_DIR>` contains the following:

* `ILSVRC2012_img_val` - directory containing the ILSVRC 2012 validation images
* `val.txt` - annotation file used for ILSVRC 2012
* `val15.txt` - annotation file used for ILSVRC 2015

### Datasets in dataset_definitions.yml
* `imagenet_1000_classes` used for evaluation models trained on ILSVRC 2012 dataset with 1000 classes. (model examples: [`alexnet`](models/public/alexnet/alexnet.md), [`vgg16`](models/public/vgg16/vgg16.md))
* `imagenet_1000_classes_2015` used for evaluation models trained on ILSVRC 2015 dataset with 1000 classes. (model examples: [`se-resnet-152`](models/public/se-resnet-152/se-resnet-152.md), [`se-resnext-50`](models/public/se-resnext-50/se-resnext-50.md))
* `imagenet_1001_classes` used for evaluation models trained on ILSVRC 2012 dataset with 1001 classes (background label + original labels). (model examples: [`googlenet-v2-tf`](models/public/googlenet-v2-tf/googlenet-v2-tf.md), [`resnet-50-tf`](models/public/resnet-50-tf/resnet-50-tf.md))

## [Common Objects in Context (COCO)](http://cocodataset.org/#home)

### How download dataset

To download COCO dataset, you need to follow the steps below:
1. Download [`2017 Val images`](http://images.cocodataset.org/zips/val2017.zip) and [`2017 Train/Val annotations`](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
2. Unpack archives

### Files layout

To use this dataset with OMZ tools, make sure `<DATASET_DIR>` contains the following:

* `val2017` - directory containing the COCO 2017 validation images
* `instances_val2017.json` - annotation file which used for object detection and instance segmentation tasks
* `person_keypoints_val2017.json` - annotation file which used for human pose estimation tasks

### Datasets in dataset_definitions.yml
* `ms_coco_mask_rcnn` used for evaluation models trained on COCO dataset for object detection and instance segmentation tasks. Background label + label map with 80 public available object categories are used. Annotations are saved in order of ascending image ID.
* `ms_coco_detection_91_classes` used for evaluation models trained on COCO dataset for object detection tasks. Background label + label map with 80 public available object categories are used (original indexing to 91 categories is preserved. You can find more information about object categories labels [here](https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/)). Annotations are saved in order of ascending image ID. (model examples: [`faster_rcnn_resnet50_coco`](models/public/faster_rcnn_resnet50_coco/faster_rcnn_resnet50_coco.md), [`ssd_resnet50_v1_fpn_coco`](models/public/ssd_resnet50_v1_fpn_coco/ssd_resnet50_v1_fpn_coco.md))
* `ms_coco_detection_80_class_with_background` used for evaluation models trained on COCO dataset for object detection tasks. Background label + label map with 80 public available object categories are used. Annotations are saved in order of ascending image ID. (model examples: [`faster-rcnn-resnet101-coco-sparse-60-0001`](models/intel/faster-rcnn-resnet101-coco-sparse-60-0001/description/faster-rcnn-resnet101-coco-sparse-60-0001.md), [`ssd-resnet34-1200-onnx`](models/public/ssd-resnet34-1200-onnx/ssd-resnet34-1200-onnx.md))
* `ms_coco_detection_80_class_without_background` used for evaluation models trained on COCO dataset for object detection tasks. Label map with 80 public available object categories is used. Annotations are saved in order of ascending image ID. (model examples: [`ctdet_coco_dlav0_384`](models/public/ctdet_coco_dlav0_384/ctdet_coco_dlav0_384.md), [`yolo-v3-tf`](models/public/yolo-v3-tf/yolo-v3-tf.md))
* `ms_coco_keypoints` used for evaluation models trained on COCO dataset for human pose estimation tasks. Each annotation stores multiple keypoints for one image. (model examples: [`human-pose-estimation-0001`](models/intel/human-pose-estimation-0001/description/human-pose-estimation-0001.md))
* `ms_coco_single_keypoints` used for evaluation models trained on COCO dataset for human pose estimation tasks. Each annotation stores single keypoints for image, so several annotation can be associated to one image. (model examples: [`single-human-pose-estimation-0001`](models/public/single-human-pose-estimation-0001/description/single-human-pose-estimation-0001.md))

## [WIDER FACE](http://shuoyang1213.me/WIDERFACE/)

### How download dataset

To download WIDER Face dataset, you need to follow the steps below:
1. Go to the [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) website
2. Go to the `Download` section
3. Select `WIDER Face Validation images` and download them from [Google Drive](https://drive.google.com/file/d/0B6eKvaijfFUDd3dIRmpvSk8tLUk/view) or [Tencent Drive](https://share.weiyun.com/5ot9Qv1)
4. Select and download `Face annotations`
5. Unpack archives

### Files layout

To use this dataset with OMZ tools, make sure `<DATASET_DIR>` contains the following:

* `WIDER_val` - directory containing images directory
    * `images` - directory containing the WIDER Face validation images
* `wider_face_split` - directory with annotation file
    * `wider_face_val_bbx_gt.txt` - annotation file

### Datasets in dataset_definitions.yml
* `wider` used for evaluation models on WIDER Face dataset where the face is the first class. (model examples: [`mtcnn`](models/public/mtcnn/mtcnn.md), [`retinaface-resnet50`](models/public/retinaface-resnet50/retinaface-resnet50.md))
* `wider_without_bkgr` used for evaluation models on WIDER Face dataset where the face is class zero. (model examples: [`mobilefacedet-v1-mxnet`](models/public/mobilefacedet-v1-mxnet/mobilefacedet-v1-mxnet.md))
