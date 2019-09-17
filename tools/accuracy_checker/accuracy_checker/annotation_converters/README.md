# Annotation Converters

Annotation converter is a function which converts annotation file to suitable for metric evaluation format.
Each annotation converter expects specific annotation file format or data structure, which depends on original dataset.
If converter for your data format is not supported by Accuracy Checker, you can provide your own annotation converter.
Each annotation converter has parameters available for configuration.

Process of conversion can be implemented in two ways:
* via configuration file
* via command line

### Describing annotation conversion in configuration file.

Annotation conversion can be provided in `dataset` section your configuration file to convert annotation inplace before every evaluation.
Each conversion configuration should contain `converter` field filled selected converter name and provide converter specific parameters (more details in supported converters section). All paths can be prefixed via command line with `-s, --source` argument.

You can additionally use optional parameters like:
* `subsample_size` - Dataset subsample size. You can specify the number of ground truth objects or dataset ratio in percentage. Please, be careful to use this option, some datasets does not support subsampling. You can also specify `subsample_seed` if you want to generate subsample with specific random seed.
* `annotation` - path to store converted annotation pickle file. You can use this parameter if you need to reuse converted annotation to avoid subsequent conversions.
* `dataset_meta` - path to store mata information about converted annotation if it is provided.
* `analyze_dataset` - flag which allow to get statistics about converted dataset. Supported annotations: `ClassificationAnnotation`, `DetectionAnnotation`, `MultiLabelRecognitionAnnotation`, `RegressionAnnotation`. Default value is False.

Example of usage:

```yaml
   annotation_conversion:
     # Converter name which will be called for conversion.
     converter: sample
     # Converter specific parameters, can be different depend on converter realization.
     data_dir: sample/sample_dataset
   # (Optional) subsample generation. Can be also used with prepared annotation file.
   subsample_size: 1000
   # (Optional) paths to store annotation files for following usage. In the next evaluation these files will be directly used instead running conversion.
   annotation: sample_dataset.pickle
   dataset_meta: sample_dataset.json
```

### Conversing process via command line.

The command line for annotation conversion looks like:

```bash
convert_annotation <converter_name> <converter_specific parameters>
```
All converter specific options should have format `--<parameter_name> <parameter_value>`
You may refer to `-h, --help` to full list of command line options. Some optional arguments are:

* `-o, --output_dir` - directory to save converted annotation and meta info.
* `-a, --annotation_name` - annotation file name.
* `-m, --meta_name` - meta info file name.

### Supported converters 

Accuracy Checker supports following list of annotation converters and specific for them parameters:
* `cifar10` - converts CIFAR 10 classification dataset to `ClassificationAnnotation`
  * `data_batch_file` - path to pickle file which contain dataset batch (e.g. test_batch)
  * `has_background` - allows to add background label to original labels and convert dataset for 11 classes instead 10 (default value is False).
  * `convert_images` - allows to convert images from pickle file to user specified directory (default value is False).
  * `converted_images_dir` - path to converted images location.
* `mnist_csv` - convert MNIST dataset for handwritten digit recognition stored in csv format to `ClassificationAnnotation`.
  * `annotation_file` - path to dataset file in csv format.
  * `convert_images` - allows to convert images from annotation file to user specified directory (default value is False).
  * `converted_images_dir` - path to converted images location if enabled `convert_images`.
* `imagenet` - convert ImageNet dataset for image classification task to `ClassificationAnnotation`.
  * `annotation_file` - path to annotation in txt format.
  * `labels_file` - path to file with word description of labels (synset_words).
  * `has_background` - allows to add background label to original labels and convert dataset for 1001 classes instead 1000 (default value is False).
* `voc_detection` - converts Pascal VOC annotation for detection task to `DetectionAnnotation`.
   * `imageset_file` - path to file with validation image list.
   * `annotations_dir` - path to directory with annotation files.
   * `images_dir` - path to directory with images related to devkit root (default JPEGImages).
  * `has_background` - allows convert dataset with/without adding background_label. Accepted values are True or False. (default is True) 
* `voc_segmentation` - converts Pascal VOC annotation for semantic segmentation task to `SegmentationAnnotation`.
  * `imageset_file` - path to file with validation image list.
  * `images_dir` - path to directory with images related to devkit root (default JPEGImages).
  * `mask_dir` - path to directory with ground truth segmentation masks related to devkit root (default SegmentationClass).
* `mscoco_detection` - converts MS COCO dataset for object detection task to `DetectionAnnotation`.
  * `annotation_file` - path ot annotation file in json format.
  * `has_background` - allows convert dataset with/without adding background_label. Accepted values are True or False. (default is False).
  * `use_full_label_map` - allows to use original label map (with 91 object categories) from paper instead public available(80 categories).
  * `sort_annotations` - allows to save annotations in image id ascend order.
* `mscoco_segmentation` - converts MS COCO dataset for object instance segmentation task to `CocoInstanceSegmentationAnnotation`.
  * `annotation_file` - path ot annotation file in json format.
  * `has_background` - allows convert dataset with/without adding background_label. Accepted values are True or False. (default is False).
  * `use_full_label_map` - allows to use original label map (with 91 object categories) from paper instead public available(80 categories).
  * `sort_annotations` - allows to save annotations in image id ascend order.
* `mscoco_mask_rcnn` - converts MS COCO dataset to `ContainerAnnotation` with `DetectionAnnotation` and `CocoInstanceSegmentationAnnotation` named `detection_annotation` and `segmentation_annotation` respectively.
  * `annotation_file` - path ot annotation file in json format.
  * `has_background` - allows convert dataset with/without adding background_label. Accepted values are True or False. (default is False).
  * `use_full_label_map` - allows to use original label map (with 91 object categories) from paper instead public available(80 categories).
  * `sort_annotations` - allows to save annotations in image id ascend order.
* `mscoco_keypoints` - converts MS COCO dataset for keypoints localization task to `PoseEstimationAnnotation`.
  * `annotation_file` - path ot annotation file in json format.
* `wider` - converts from Wider Face dataset to `DetectionAnnotation`.
  * `annotation_file` - path to txt file, which contains ground truth data in WiderFace dataset format.
  * `label_start` - specifies face label index in label map. Default value is 1. You can provide another value, if you want to use this dataset for separate label validation,
  in case when your network predicts other class for faces.
* `detection_opencv_storage` - converts detection annotation stored in Detection OpenCV storage format to `DetectionAnnotation`.
  * `annotation_file` - path to annotation in xml format.
  * `image_names_file` - path to txt file, which contains image name list for dataset.
  * `label_start` - specifies label index start in label map. Default value is 1. You can provide another value, if you want to use this dataset for separate label validation.
  * `background_label` - specifies which index will be used for background label. You can not provide this parameter if your dataset has not background label.
* `cityscapes` - converts CityScapes Dataset to `SegmentationAnnotation`.
  * `dataset_root_dir` - path to dataset root.
  * `images_subfolder` - path from dataset root to directory with validation images (Optional, default `imgsFine/leftImg8bit/val`).
  * `masks_subfolder` - path from dataset root to directory with ground truth masks (Optional, `gtFine/val`).
  * `masks_suffix` - suffix for mask file names (Optional, default `_gtFine_labelTrainIds`).
  * `images_suffix` - suffix for image file names (Optional, default `_leftImg8bit`).
  * `use_full_label_map` - allows to use full label map with 33 classes instead train label map with 18 classes (Optional, default `False`).
* `vgg_face` - converts VGG Face 2 dataset for facial landmarks regression task to `FacialLandmarksAnnotation`.
  * `landmarks_csv_file` - path to csv file with coordinates of landmarks points.
  * `bbox_csv_file` - path to cvs file which contains bounding box coordinates for faces (optional parameter).
* `lfw` - converts Labeled Faces in the Wild dataset for face reidentification to `ReidentificationClassificationAnnotation`.
  * `pairs_file` - path to file with annotation positive and negative pairs.
  * `train_file` - path to file with annotation positive and negative pairs used for network train (optional parameter).
  * `landmarks_file` - path to file with facial landmarks coordinates for annotation images (optional parameter).
* `mars` - converts MARS person reidentification dataset to `ReidentificationAnnotation`.
  * `data_dir` - path to data directory, where gallery (`bbox_test`) and `query` subdirectories are located.
* `market1501_reid` - converts Market1501 person reidentification dataset to `ReidentificationAnnotation`.
  * `data_dir` - path to data directory, where gallery (`bounding_box_test`) and `query` subdirectories are located.
* `super_resolution` - converts dataset for super resolution task to `SuperResolutionAnnotation`.
  * `data_dir` - path to folder, where images in low and high resolution are located.
  * `lr_suffix` - low resolution file name's suffix (default lr).
  * `hr_suffix` - high resolution file name's suffix (default hr).
  * `annotation_loader` - which library will be used for ground truth image reading. Supported: `opencv`, `pillow` (Optional. Default value is pillow). Note, color space of image depends on loader (OpenCV uses BGR, Pillow uses RGB for image reading).
* `icdar_detection` - converts ICDAR13 and ICDAR15 datasets for text detection challenge to `TextDetectionAnnotation`.
  * `data_dir` - path to folder with annotations on txt format.
* `icdar13_recognition` - converts ICDAR13 dataset for text recognition task to `CharecterRecognitionAnnotation`.
  * `annotation_file` - path to annotation file in txt format.
* `brats` - converts BraTS dataset format to `BrainTumorSegmentationAnnotation` format.
  * `data_dir` - dataset root directory, which contain subdirectories with validation data (`imagesTr`) and ground truth labels (`labelsTr`).
  Optionally you can provide relative path for these subdirectories (if they have different location) using `image_folder` and `mask_folder` parameters respectively.
* `movie_lens_converter` - converts Movie Lens Datasets format to `HitRatioAnnotation` format.
  * `rating_file` - path to file which contains movieId with top score for each userID (for example ml-1m-test-ratings.csv)
  * `negative_file` - path to file which contains negative examples.
  * `users_max_number` - the number of users which will be used for validation (Optional, it gives opportunity to cut list of users. If argument is not provided, full list of users will be used.).
* `brats_numpy` - converts Brain Tumor Segmentation dataset to `BrainTumorSegmentationAnnotation`. This converter works with Numpy representation of BraTS dataset.
  * `data_dir` - path to dataset root directory.
  * `ids_file` - path to file, which contains names of images in dataset
  * `labels_file` - path to file, which contains labels (optional, if omitted no labels will be shown)
  * `data_suffix` - suffix for files with data (default `_data_cropped`)
  * `label_suffix` - suffix for files with groundtruth data (default `_label_cropped`)
  * `boxes_file` - path to file with brain boxes (optional). Set this option with including postprocessor `segmentation-prediction-resample`(see [Postprocessors](../postprocessor/README.md)).
* `wmt` - converts WMT dataset for Machine Translation task to `MachineTranslationAnnotation`.
  * `input_file` - path to file which contains input sentences tokens for translation.
  * `reference_file` - path to file with reference for translation.
* `common_semantic_segmentation` - converts general format of datasets for semantic segmentation task to `SegmentationAnnotation`. The converter expects following dataset structure:
  1. images and GT masks are located in separated directories (e.g. `<dataset_root>/images` for images and `<dataset_root>/masks` for masks respectively)
  2. images and GT masks has common part in names and can have difference in prefix and postfix (e.g. image name is image0001.jpeg, mask for it is gt0001.png are acceptable. In this case base_part - 0001, image_prefix - image, image_postfix - .jpeg, mask_prefix - gt, mask_postfix - .png)
  * `images_dir` - path to directory with images.
  * `masks_dir` - path to directory with GT masks.
  * `image_prefix` - prefix part for image file names. (Optional, default is empty).
  * `image_postfix` - postfix part for image file names (optional, default is `.png`).
  * `mask_prefix` - prefix part for mask file names. (Optional, default is empty).
  * `image_postfix` - postfix part for mask file names (optional, default is `.png`).
  * `mask_loader` - the way how GT mask should be loaded. Supported methods: `pillow`, `opencv`, `nifti`, `numpy`, `scipy`.
  * `dataset_meta` - path to json file with prepared dataset meta info. It should contains `label_map` key with dictionary in format class_id: class_name and optionally `segmentation_colors` (if your dataset uses color encoding). Segmentation colors is a list of channel-wise values for each class. (e.g. if your dataset has 3 classes in BGR colors, segmentation colors for it will looks like: `[[255, 0, 0], [0, 255, 0], [0, 0, 255]]`). (Optional, you can provide self-created file as `dataset_meta` in your config).
* `camvid` - converts CamVid dataset format to `SegmentationAnnotation`.
  * `annotation_file` - file in txt format which contains list of validation pairs (`<path_to_image>` `<path_to_annotation>` separated by space)
* `image_retrieval` - converts dataset for image retrieval task to `ReidentificationAnnotation`. Dataset should have following structure:
   1. the dataset root directory contains 2 subdirectory named `gallery` and `queries` for gallery images and query images respectively.
   2. Every of these subdirectories should contains text file with list of pairs: `<path_to_image>` `<image_ID>` (image_path and image_ID should be separated by space),  where `<path_to_image>` is path to the image related dataset root, `<image_ID>` is the number which represent image id in the gallery.
   * `data_dir` - path to dataset root directory.
   * `gallery_annotation_file` - file with gallery images and IDs concordance in txt format (Optional, default value is `<data_dir>/gallery/list.txt`)
   * `queries_annotation_file` - file with queries images and IDs concordance in txt format (Optional, default value is `<data_dir>/queries/list.txt`)
* `cvat_object_detection` - converts [CVAT XML annotation version 1.1](https://github.com/opencv/cvat/blob/develop/cvat/apps/documentation/xml_format.md#xml-annotation-format) format for images to `DetectionAnnotation`.
  * `annotation_file` - path to xml file in appropriate format.
  * `has_background` - allows prepend original labels with special class represented background and convert dataset for n+1 classes instead n (default value is True).
* `cvat_attributes_recognition` - converts [CVAT XML annotation version 1.1](https://github.com/opencv/cvat/blob/develop/cvat/apps/documentation/xml_format.md#xml-annotation-format) format for images to `ClassificationAnnotation` or `ContainerAnnotation` with `ClassificationAnnotation` as value type and attribute names as keys (in multiple attributes case). Used bbox attributes as annotation classes.
  * `annotation_file` - path to xml file in appropriate format.
  * `label` - the dataset label which will be used for attributes collection (e.g. if your dataset contains 2 labels: `face` and `person` and you want recognise attributes for face, you should use `face` as value for this parameter).
* `cvat_age_gender` -  converts [CVAT XML annotation version 1.1](https://github.com/opencv/cvat/blob/develop/cvat/apps/documentation/xml_format.md#xml-annotation-format) format for images which represent dataset for age gender recognition to `ContainerAnnotation` with `ClassificationAnnotation` for gender recognition, `ClassificationAnnotation` for age classification and `RegeressionAnnotation` for age regression. The identifiers for representations following: `gender_annotation`, `age_class_annotation`, `age_regression_annotation`.
  * `annotation_file` - path to xml file in appropriate format.
* `cvat_facial_landmarks` - converts [CVAT XML annotation version 1.1](https://github.com/opencv/cvat/blob/develop/cvat/apps/documentation/xml_format.md#xml-annotation-format) format for images to `FacialLandmarksAnnotation`.
  * `annotation_file` - path to xml file in appropriate format.
* `cvat_pose_estimation` - converts [CVAT XML annotation version 1.1](https://github.com/opencv/cvat/blob/develop/cvat/apps/documentation/xml_format.md#xml-annotation-format) format for images to `PoseEstimationAnnotation`.
  * `annotation_file` - path to xml file in appropriate format.
* `cvat_text_recognition` - converts [CVAT XML annotation version 1.1](https://github.com/opencv/cvat/blob/develop/cvat/apps/documentation/xml_format.md#xml-annotation-format) format for images to `CharacterRecognitionAnnotation`.
  * `annotation_file` - path to xml file in appropriate format.
* `cvat_binary_multilabel_attributes_recognition` - converts [CVAT XML annotation version 1.1](https://github.com/opencv/cvat/blob/develop/cvat/apps/documentation/xml_format.md#xml-annotation-format) format for images to `MultiLabelRecognitionAnnotation`. Used bbox attributes as annotation classes. Each attribute field should contains `T` or `F` values for attribute existence/non-existence on the image respectively.
  * `annotation_file` - path to xml file in appropriate format.
  * `label` - the dataset label which will be used for attributes collection (e.g. if your dataset contains 2 labels: `face` and `person` and you want recognise attributes for face, you should use `face` as value for this parameter).
* `cvat_person_detection_action_recognition` converts dataset with [CVAT XML annotation version 1.1](https://github.com/opencv/cvat/blob/develop/cvat/apps/documentation/xml_format.md#xml-annotation-format) for person detection and action recognition task to `ContainerAnnotation` with `DetectionAnnotation` for person detection quality estimation named `person_annotation` and `ActionDetectionAnnotation` for action recognition named `action_annotation`.
  * `annotation_file` - path to xml file with ground truth.
  * `use_case` - use case, which determines the dataset label map. Supported range actions:
    * `common_3_actions`(seating, standing, raising hand)
    * `common_6_actions`(seating, writing, raising hand, standing, turned around, lie on the desk)
    * `teacher` (standing, writing, demonstrating)
    * `raising_hand` (seating, raising hand)
* `lpr_txt` - converts annotation for license plate recognition task in txt format to `CharacterRecognitionAnnotation`.
  * `annotation_file` - path to txt annotation.
  * `decoding_dictionary` - path to file containing dictionary for output decoding.
