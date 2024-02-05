# Dataset Preparation Guide

If you want to use prepared configs to run the Accuracy Checker tool, you need to organize `<DATASET_DIR>` folder with validation datasets in a certain way. Instructions for preparing validation data are described in this document.

Each dataset description consists of the following sections:
* instruction for downloading the dataset
* structure of `<DATASET_DIR>` that matches the dataset definition in the existing global configuration file (`<omz_dir>/data/dataset_definitions.yml`)
* examples of using and presenting the dataset in the global configuration file

More detailed information about using predefined configuration files you can find [here](../tools/accuracy_checker/configs/README.md).

## ImageNet

### Download dataset

To download images from ImageNet, you need to have an account and agree to the Terms of Access.
1. Go to the [ImageNet home page](http://www.image-net.org/).
2. If you have an account, click `Login`. Otherwise, click `Signup` in the right upper corner, provide your data, and wait for a confirmation email.
3. Log in after receiving the confirmation email and go to the `Download` tab.
4. Select `Download Original Images`.
5. You will be redirected to the Terms of Access page. If you agree to the Terms, continue by clicking Agree and Sign.
6. Click one of the links in the `Download as one tar file` section to select it.
7. Unpack archive.

To download annotation files:
* `val.txt`
  1. Download [archive](http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz)
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
* `imagenet_1000_classes` used for evaluation models trained on ILSVRC 2012 dataset with 1000 classes. (model example: [densenet-121-tf](../models/public/densenet-121-tf/README.md))
* `imagenet_1000_classes_2015` used for evaluation models trained on ILSVRC 2015 dataset with 1000 classes.
* `imagenet_1001_classes` used for evaluation models trained on ILSVRC 2012 dataset with 1001 classes (background label + original labels). (model examples: [googlenet-v2-tf](../models/public/googlenet-v2-tf/README.md), [resnet-50-tf](../models/public/resnet-50-tf/README.md))

## Common Objects in Context (COCO)

### Download dataset

1. Go to the [COCO home page](https://cocodataset.org/#home).
2. Click `Dataset` in the menu and select `Download`.
3. Download [2017 Val images](http://images.cocodataset.org/zips/val2017.zip) and [2017 Train/Val annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip).
4. Unpack archives.

### Files layout

To use this dataset with OMZ tools, make sure `<DATASET_DIR>` contains the following:

* `val2017` - directory containing the COCO 2017 validation images
* `annotations` - directory containing the COCO 2017 annotation files
  * `instances_val2017.json` - annotation file which used for object detection and instance segmentation tasks
  * `person_keypoints_val2017.json` - annotation file which used for human pose estimation tasks

### Datasets in dataset_definitions.yml
* `ms_coco_mask_rcnn` used for evaluation models trained on COCO dataset for object detection and instance segmentation tasks. Background label + label map with 80 public available object categories are used. Annotations are saved in order of ascending image ID.
* `ms_coco_detection_91_classes` used for evaluation models trained on COCO dataset for object detection tasks. Background label + label map with 80 public available object categories are used (original indexing to 91 categories is preserved. You can find more information about object categories labels [here](https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/)). Annotations are saved in order of ascending image ID. (model examples: [faster_rcnn_resnet50_coco](../models/public/faster_rcnn_resnet50_coco/README.md), [ssd_mobilenet_v1_coco](../models/public/ssd_mobilenet_v1_coco/README.md))
* `ms_coco_detection_80_class_with_background` used for evaluation models trained on COCO dataset for object detection tasks. Background label + label map with 80 public available object categories are used. Annotations are saved in order of ascending image ID. (model examples: [faster-rcnn-resnet101-coco-sparse-60-0001](../models/intel/faster-rcnn-resnet101-coco-sparse-60-0001/README.md), [ssd-resnet34-1200-onnx](../models/public/ssd-resnet34-1200-onnx/README.md))
* `ms_coco_detection_80_class_without_background` used for evaluation models trained on COCO dataset for object detection tasks. Label map with 80 public available object categories is used. Annotations are saved in order of ascending image ID. (model examples: [ctdet_coco_dlav0_512](../models/public/ctdet_coco_dlav0_512/README.md), [yolo-v3-tf](../models/public/yolo-v3-tf/README.md))
* `ms_coco_keypoints` used for evaluation models trained on COCO dataset for human pose estimation tasks. Each annotation stores multiple keypoints for one image. (model examples: [human-pose-estimation-0001](../models/intel/human-pose-estimation-0001/README.md))
* `ms_coco_single_keypoints` used for evaluation models trained on COCO dataset for human pose estimation tasks. Each annotation stores single keypoints for image, so several annotation can be associated to one image. (model examples: [single-human-pose-estimation-0001](../models/public/single-human-pose-estimation-0001/README.md))

## WIDER FACE

### Download dataset

1. Go to the [WIDER FACE home page](http://shuoyang1213.me/WIDERFACE/).
2. Go to the `Download` section.
3. Select `WIDER Face Validation images` and download them from [Google Drive](https://drive.google.com/file/d/0B6eKvaijfFUDd3dIRmpvSk8tLUk/view) or [Tencent Drive](https://share.weiyun.com/5ot9Qv1).
4. Select and download `Face annotations`.
5. Unpack archives.

### Files layout

To use this dataset with OMZ tools, make sure `<DATASET_DIR>` contains the following:

* `WIDER_val` - directory containing images directory
    * `images` - directory containing the WIDER Face validation images
* `wider_face_split` - directory with annotation file
    * `wider_face_val_bbx_gt.txt` - annotation file

### Datasets in dataset_definitions.yml
* `wider` used for evaluation models on WIDER Face dataset where the face is the first class. (model example: [faceboxes-pytorch](../models/public/faceboxes-pytorch/README.md))
* `wider_without_bkgr` used for evaluation models on WIDER Face dataset where the face is class zero. (model example: [face-detection-0204](../models/intel/face-detection-0204/README.md))

## Visual Object Classes Challenge 2012 (VOC2012)

### Download dataset

1. Go to the [VOC2012 website](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).
2. Go to the [Development Kit](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit) section.
3. Click [Download the training/validation data](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) to download archive.
4. Unpack archive.

### Files layout

To use this dataset with OMZ tools, make sure `<DATASET_DIR>` contains the following:

* `VOCdevkit/VOC2012` - directory containing annotations, images, segmentation masks and image sets files directories
    * `Annotations` - directory containing the VOC2012 annotation files
    * `JPEGImages` - directory containing the VOC2012 validation images
    * `ImageSets` - directory containing the VOC2012 text files specifying lists of images for different tasks
        * `Main/val.txt` - image sets file for detection tasks
        * `Segmentation/val.txt` - image sets file for segmentation tasks
    * `SegmentationClass` - directory containing the VOC2012 segmentation masks

### Datasets in dataset_definitions.yml
* `VOC2012` used for evaluation models on VOC2012 dataset for object detection task. Background label + label map with 20 object categories are used.
* `VOC2012_without_background` used for evaluation models on VOC2012 dataset for object detection tasks. Label map with 20 object categories is used.(model examples: [yolo-v2-ava-0001](../models/intel/yolo-v2-ava-0001/README.md), [yolo-v2-tiny-ava-0001](../models/intel/yolo-v2-tiny-ava-0001/README.md))
* `VOC2012_Segmentation` used for evaluation models on VOC2012 dataset for segmentation tasks. Background label + label map with 20 object categories are used.(model examples: [deeplabv3](../models/public/deeplabv3/README.md))

## Visual Object Classes Challenge 2007 (VOC2007)

### Download dataset

1. Go to the [VOC2007 website](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/).
2. Go to the [Development Kit](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/#devkit) section.
3. Click [Download the training/validation data](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) to download archive.
4. Unpack archive.

### Files layout

To use this dataset with OMZ tools, make sure `<DATASET_DIR>` contains the following:

* `VOCdevkit/VOC2007` - directory containing annotations, images and image sets files directories
    * `Annotations` - directory containing the VOC2007 annotation files
    * `JPEGImages` - directory containing the VOC2007 images
    * `ImageSets` - directory containing the VOC2007 text files specifying lists of images for different tasks
        * `Main/test.txt` - image sets file for detection tasks

### Datasets in dataset_definitions.yml
* `VOC2007_detection` used for evaluation models on VOC2007 dataset for object detection task. Background label + label map with 20 object categories are used. (model example: [yolo-v1-tiny-tf](../models/public/yolo-v1-tiny-tf/README.md))
* `VOC2007_detection_no_bkgr` used for evaluation models on VOC2007 dataset for object detection tasks. Label map with 20 object categories is used.(model example: [yolo-v1-tiny-tf](../models/public/yolo-v1-tiny-tf/README.md))

## SYGData0829

### Download dataset

1. Go to the [SYGData0829 Github repository](https://github.com/ermubuzhiming/OMZ-files-download/releases/tag/v1-ly).
2. Select:
    - [SYGData0829.z01](https://github.com/ermubuzhiming/OMZ-files-download/releases/download/v1-ly/SYGData0829.z01),
    - ['SYGData0829.z02'](https://github.com/ermubuzhiming/OMZ-files-download/releases/download/v1-ly/SYGData0829.z02),
    - ['SYGData0829.z03'](https://github.com/ermubuzhiming/OMZ-files-download/releases/download/v1-ly/SYGData0829.z03),
    - ['SYGData0829.zip'](https://github.com/ermubuzhiming/OMZ-files-download/releases/download/v1-ly/SYGData0829.zip).
3. Unpack archive.

### Files layout
* `SYGData0829/dataset_format_VOC2007` - directory containing annotations, images and image sets files directories
    * `Annotations` - directory containing the SYGData0829 annotation files
    * `JPEGImages` - directory containing the SYGData0829 images
    * `ImageSets` - directory containing the SYGData0829 text files specifying lists of images for different tasks
		* `Main/val.txt` - image sets file for validation of detection tasks

### Datasets in dataset_definitions.yml
* `SYGData0829` used for evaluation models on SYGData0829 dataset for object detection task. Label map with 4 object categories are used. (model examples: [mobilenet-yolo-v4-syg](../models/public/mobilenet-yolo-v4-syg/README.md))

## [erfnet_data](https://github.com/Zhangxianwen2021/ERFNet/releases/tag/erfnet)

### How to download dataset

To download erfnet_data dataset, you need to follow the steps below:
1. Go to the [github repo](https://github.com/Zhangxianwen2021/ERFNet/releases/tag/erfnet)
2. Select [`Annotations.rar`](https://github.com/Zhangxianwen2021/ERFNet/releases/download/erfnet/Annotations.rar)
Select ['JPEGImages.rar'](https://github.com/Zhangxianwen2021/ERFNet/releases/download/erfnet/JPEGImages.rar)
Select ['erfnet_meta_zxw.json'](https://github.com/Zhangxianwen2021/ERFNet/releases/download/erfnet/erfnet_meta_zxw.json)
Select ['val.txt'](https://github.com/Zhangxianwen2021/ERFNet/releases/download/erfnet/val.txt)
3. Unpack archive

### Files layout
* `erfnet_data` - directory containing annotations, images, image sets and dataset meta files directories
    * `Annotations` - directory containing the erfnet_data annotation files
    * `JPEGImages` - directory containing the erfnet_data images
    * `erfnet_meta_zxw.json` - directory containing the erfnet_data text files specifying lists of images for different tasks
    * `val.txt` - image sets file for validation of detection tasks

### Datasets in dataset_definitions.yml
* `erfnet_data` used for evaluation models on erfnet_data dataset for object segmentation task. (model examples: [`erfnet`](../models/public/erfnet/README.md))

## PASCAL-S

### Download dataset

1. Go to the [The Secrets of Salient Object Segmentation home page](http://cbs.ic.gatech.edu/salobj/).
2. Go to the `Download` section.
3. Click [Dataset & Code](http://cbs.ic.gatech.edu/salobj/download/salObj.zip) to download archive.
4. Unpack archive.

### Files layout

To use this dataset with OMZ tools, make sure `<DATASET_DIR>` contains the following:

* `PASCAL-S` - directory containing images and salient region masks subdirectories
    * `image` - directory containing the PASCAL-S images from the directory `datasets/imgs/pascal` in the unpacked archive
    * `mask` - directory containing the PASCAL-S salient region masks from the directory `datasets/masks/pascal` in the unpacked archive

### Datasets in dataset_definitions.yml
* `PASCAL-S` used for evaluation models on PASCAL-S dataset for salient object detection task. (model examples: [f3net](../models/public/f3net/README.md))

## CoNLL2003 Named Entity Recognition

See the [CoNLL2003 Named Entity Recognition website](https://www.aclweb.org/anthology/W03-0419/).

### Download dataset

1. Download [archive](https://data.deepai.org/conll2003.zip) from the [CoNLL 2003 (English) Dataset](https://deepai.org/dataset/conll-2003-english) page in the [DeepAI Datasets](https://deepai.org/datasets) website.
2. Unpack archive.

### Files layout

To use this dataset with OMZ tools, make sure `<DATASET_DIR>` contains the following:
* `CONLL-2003` - directory containing annotation files
    * `valid.txt` - annotation file for CoNLL2003 validation set

### Datasets in dataset_definitions.yml
* `CONLL2003_bert_cased` used for evaluation models on CoNLL2003 dataset for named entity recognition task. (model examples: [bert-base-ner](../models/public/bert-base-ner/README.md))

## MRL Eye

### Download dataset

1. Go to the [MRL Eye Dataset website](http://mrl.cs.vsb.cz/eyedataset).
2. Download [archive](http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip).
3. Unpack archive.

### Files layout

To use this dataset with OMZ tools, make sure `<DATASET_DIR>` contains the following:

* `mrlEyes_2018_01` - directory containing subdirectories with dataset images

### Datasets in dataset_definitions.yml
* `mrlEyes_2018_01` used for evaluation models on MRL Eye dataset for recognition of eye state. (model examples: [open-closed-eye-0001](../models/public/open-closed-eye-0001/README.md))

## Labeled Faces in the Wild (LFW)

### Download dataset

1. Go to the [Labeled Faces in the Wild home page](http://vis-www.cs.umass.edu/lfw/).
2. Go to the `Download the database` section.
3. Click `All images as gzipped tar file` to download archive.
4. Unpack archive.
5. Go to the `Training, Validation, and Testing` section.
6. Select `pairs.txt` and download pairs file.
7. Download [lfw_landmark](https://raw.githubusercontent.com/clcarwin/sphereface_pytorch/master/data/lfw_landmark.txt) file.

### Files layout

To use this dataset with OMZ tools, make sure `<DATASET_DIR>` contains the following:

* `LFW` - directory containing images directories, pairs and landmarks files
    * `lfw` - directory containing the LFW images
    * `annotation` - directory containing pairs and landmarks files
        * `pairs.txt` - file with annotation positive and negative pairs for LFW dataset
        * `lfw_landmark.txt` - file with facial landmarks coordinates for annotation images of LFW dataset

### Datasets in dataset_definitions.yml
* `lfw` used for evaluation models on LFW dataset for face recognition task. (model example: [face-reidentification-retail-0095](../models/intel/face-reidentification-retail-0095/README.md))

## NYU Depth Dataset V2

See the the [NYU Depth Dataset V2 website](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html).

### Download dataset

To download NYU Depth Dataset V2 preprocessed data stored in HDF5 format:
1. Download [archive](http://datasets.lids.mit.edu/fastdepth/data/nyudepthv2.tar.gz) from the [website](http://datasets.lids.mit.edu/fastdepth/data/).
2. Unpack archive.

### Files layout

To use this dataset with OMZ tools, make sure `<DATASET_DIR>` contains the following:

* `nyudepthv2/val` - directory with dataset official data and converted images and depth map
    * `official` - directory with data stored in original hdf5 format
    * `converted` - directory with converted data
        * `images` - directory with converted images
        * `depth` -  directory with depth maps

Note: If dataset is used in the first time, set `allow_convert_data: True` in annotation conversion parameters for this dataset in `dataset_definitions.yml`  or use `convert_annotation` command line interface:

```sh
convert_annotation nyu_depth_v2 --data_dir <DATASET_DIR>/nyudepthv2/val/official --allow_convert_data True
```

### Datasets in dataset_definitions.yml
* `NYU_Depth_V2` used for evaluation models on NYU Depth Dataset V2 for monocular depth estimation task. (model examples: [fcrn-dp-nyu-depth-v2-tf](../models/public/fcrn-dp-nyu-depth-v2-tf/README.md))
