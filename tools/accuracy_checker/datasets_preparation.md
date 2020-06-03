# Datasets preparation

If you want to use prepared configs to run the Accuracy Checker tool and the Model quantizer, you need to organize folders with validation datasets in a certain way.  Instructions for preparing validation data are described in this document.

## [ImageNet](http://image-net.org)

### How download dataset

To download images from ImageNet, you need to have an account and agree to the Terms of Access. Follow the steps below:
1. Go to the [ImageNet](http://www.image-net.org/) homepage
2. If you have an account, click `Login`. Otherwise, click `Signup` in the right upper corner, provide your data, and wait for a confirmation email
3. Log in after receiving the confirmation email and go to the `Download` tab
4. Select `Download Original Images`
5. You will be redirected to the Terms of Access page. If you agree to the Terms, continue by clicking Agree and Sign
6. Click one of the links in the Download as one tar file section to select it

### Structure of directory with dataset

This section describes what structure of directory dataset must have to run the Accuracy Checker tool.
`<DATASET_DIR>` must contain the following entries:
* A subdirectory named `ILSVRC2012_img_val` containing the ILSVRC 2012 validation images. How to download them is described above.
* One of the annotation files:
    * `val.txt` from <http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz>.
    * `ILSVRC2017_val.txt` from <https://raw.githubusercontent.com/hujie-frank/SENet/master/ILSVRC2017_val.txt> which must be renamed to `val15.txt`

## [Common Objects in Context (COCO)](http://cocodataset.org/#home)

### How download dataset

To download COCO dataset, you need to follow the steps below:
1. Go to the [COCO](http://cocodataset.org/#home) website
2. Go to the `Dataset-> Download` tab
3. Select [`2017 Val images`](http://images.cocodataset.org/zips/val2017.zip) and [`2017 Train/Val annotations`](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) and download these archives
4. Unpack archives

### Structure of directory with dataset

This section describes what structure of directory dataset must have to run the Accuracy Checker tool.
`<DATASET_DIR>` must contain the following entries:
* A subdirectory named `val2017` containing the COCO 2017 validation images.
* One or set of annotation files:
    * `instances_val2017.json` which used for object detection and instance segmentation tasks
    * `person_keypoints_val2017.json` which used for human pose estimation tasks

## [WIDER Face](http://shuoyang1213.me/WIDERFACE/)

### How download dataset

To download WIDER Face dataset, you need to follow the steps below:
1. Go to the [WIDER Face](http://shuoyang1213.me/WIDERFACE/) website
2. Go to the `Download` section
3. Select `WIDER Face Validation images` and download them from [Google Drive](https://drive.google.com/file/d/0B6eKvaijfFUDd3dIRmpvSk8tLUk/view) or [Tencent Drive](https://share.weiyun.com/5ot9Qv1)
4. Select and download `Face annotations`

### Structure of directory with dataset

This section describes what structure of directory dataset must have to run the Accuracy Checker tool.
`<DATASET_DIR>` must contain the following entries:
* A subdirectories named `WIDER_val/images` containing the WIDER Face validation images.
* A subdirectory `wider_face_split` with annotation file `wider_face_val_bbx_gt.txt`
