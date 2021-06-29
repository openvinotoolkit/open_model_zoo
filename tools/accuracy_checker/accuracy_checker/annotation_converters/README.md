# Annotation Converters

Annotation converter is a function which converts annotation file to suitable for metric evaluation format.
Each annotation converter expects specific annotation file format or data structure, which depends on original dataset.
If converter for your data format is not supported by Accuracy Checker, you can provide your own annotation converter.
Each annotation converter has parameters available for configuration.

Process of conversion can be implemented in two ways:
* via configuration file
* via command line

## Describing Annotation Conversion in Configuration File

Annotation conversion can be provided in `dataset` section of your configuration file to convert annotation in-place before every evaluation.
Each conversion configuration should contain `converter` field filled with a selected converter name and provide converter specific parameters (more details in supported converters section). All paths can be prefixed via command line with `-s, --source` argument.

You can additionally use optional parameters like:
* `subsample_size` - Dataset subsample size. You can specify the number of ground truth objects or dataset ratio in percentage. Please, be careful to use this option, some datasets does not support subsampling. You can also specify `subsample_seed` if you want to generate subsample with specific random seed.
* `annotation` - path to store converted annotation pickle file. You can use this parameter if you need to reuse converted annotation to avoid subsequent conversions.
* `dataset_meta` - path to store meta information about converted annotation if it is provided.
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

## Conversing Process via Command Line

The command line for annotation conversion looks like:

```bash
convert_annotation <converter_name> <converter_specific parameters>
```
All converter specific options should have format `--<parameter_name> <parameter_value>`
You may refer to `-h, --help` to full list of command line options. Some optional arguments are:

* `-o, --output_dir` - directory to save converted annotation and meta info.
* `-a, --annotation_name` - annotation file name.
* `-m, --meta_name` - meta info file name.

## Supported Converters

Accuracy Checker supports following list of annotation converters and specific for them parameters:
* `cifar` - converts [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) classification dataset to `ClassificationAnnotation`
  * `data_batch_file` - path to pickle file which contain dataset batch (e.g. test_batch)
  * `has_background` - allows to add background label to original labels (Optional, default value is False).
  * `convert_images` - allows to convert images from pickle file to user specified directory (default value is False).
  * `converted_images_dir` - path to converted images location.
  * `num_classes` - the number of classes in the dataset - 10 or 100 (Optional, default 10)
  * `dataset_meta_file` - path to json file with dataset meta (e.g. label_map, color_encoding).Optional, more details in [Customizing dataset meta](#customizing-dataset-meta) section.
* `mnist` - convert MNIST dataset for handwritten digit recognition to `ClaassificationAnnotation`. Dataset can be downloaded [here](https://deepai.org/dataset/mnist).
  * `labels_file` - binary file which contains labels.
  * `images_file` - binary file which contains images.
  * `convert_images` - allows to convert images from data file to user specified directory (default value is False).
  * `converted_images_dir` - path to converted images location if enabled `convert_images`.
  * `dataset_meta_file` - path to json file with dataset meta (e.g. label_map, color_encoding). Optional, more details in [Customizing dataset meta](#customizing-dataset-meta) section.
* `mnist_csv` - convert MNIST dataset for handwritten digit recognition stored in csv format to `ClassificationAnnotation`.
  * `annotation_file` - path to dataset file in csv format.
  * `convert_images` - allows to convert images from annotation file to user specified directory (default value is False).
  * `converted_images_dir` - path to converted images location if enabled `convert_images`.
  * `dataset_meta_file` - path to json file with dataset meta (e.g. label_map, color_encoding). Optional, more details in [Customizing dataset meta](#customizing-dataset-meta) section.
* `fashion_mnist` - convert [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset to `ClassificationAnnotation`.
  * `annotation_file` - path to labels file in binary format.
  * `data_file` - path to images file in binary format.
  * `convert_images` - allows to convert images from data file to user specified directory (default value is False).
  * `converted_images_dir` - path to converted images location if enabled `convert_images`.
  * `dataset_meta_file` - path to json file with dataset meta (e.g. label_map, color_encoding). Optional, more details in [Customizing dataset meta](#customizing-dataset-meta) section.
* `imagenet` - convert ImageNet dataset for image classification task to `ClassificationAnnotation`.
  * `annotation_file` - path to annotation in txt format.
  * `labels_file` - path to file with word description of labels (synset_words).
  * `has_background` - allows to add background label to original labels and convert dataset for 1001 classes instead 1000 (default value is False).
  * `dataset_meta_file` - path to json file with dataset meta (e.g. label_map, color_encoding).Optional, more details in [Customizing dataset meta](#customizing-dataset-meta) section.
* `voc_detection` - converts Pascal VOC annotation for detection task to `DetectionAnnotation`.
  * `imageset_file` - path to file with validation image list.
  * `annotations_dir` - path to directory with annotation files.
  * `images_dir` - path to directory with images related to devkit root (default JPEGImages).
  * `has_background` - allows convert dataset with/without adding background_label. Accepted values are True or False. (default is True)
  * `dataset_meta_file` - path to json file with dataset meta (e.g. label_map, color_encoding).Optional, more details in [Customizing dataset meta](#customizing-dataset-meta) section.
* `voc_segmentation` - converts Pascal VOC annotation for semantic segmentation task to `SegmentationAnnotation`.
  * `imageset_file` - path to file with validation image list.
  * `images_dir` - path to directory with images related to devkit root (default JPEGImages).
  * `mask_dir` - path to directory with ground truth segmentation masks related to devkit root (default SegmentationClass).
  * `dataset_meta_file` - path to json file with dataset meta (e.g. label_map, color_encoding).Optional, more details in [Customizing dataset meta](#customizing-dataset-meta) section.
**Note**: Since OpenVINO 2020.4 the converter behaviour changed. `data_source` parameter of dataset should contains directory for images only, if you have segmentation mask in separated location, please use `segmentation_masks_source` for specifying gt masks location.
* `mscoco_detection` - converts MS COCO dataset for object detection task to `DetectionAnnotation`.
  * `annotation_file` - path to annotation file in json format.
  * `has_background` - allows convert dataset with/without adding background_label. Accepted values are True or False. (default is False).
  * `use_full_label_map` - allows to use original label map (with 91 object categories) from paper instead public available(80 categories).
  * `sort_annotations` - allows to save annotations in a specific order: ascending order of image id or ascending order of image size.
  * `sort_key` - key by which annotations will be sorted(supported keys are `image_id` and `image_size`, default is `image_id`).
  * `dataset_meta_file` - path to json file with dataset meta (e.g. label_map, color_encoding).Optional, more details in [Customizing dataset meta](#customizing-dataset-meta) section.
  * `convert_COCO_to_VOC_labels` - allows to convert COCO labels to Pacsal VOC labels. Optional, default is False.
* `mscoco_segmentation` - converts MS COCO dataset for object instance segmentation task to `CocoInstanceSegmentationAnnotation`.
  * `annotation_file` - path to annotation file in json format.
  * `has_background` - allows convert dataset with/without adding background_label. Accepted values are True or False. (default is False).
  * `use_full_label_map` - allows to use original label map (with 91 object categories) from paper instead public available(80 categories).
  * `sort_annotations` - allows to save annotations in a specific order: ascending order of image id or ascending order of image size.
  * `sort_key` - key by which annotations will be sorted (supported keys are `image_id` and `image_size`, default is `image_id`).
  * `dataset_meta_file` - path to json file with dataset meta (e.g. label_map, color_encoding).Optional, more details in [Customizing dataset meta](#customizing-dataset-meta) section.
  * `semantic_only` - converts MS COCO dataset annotation to `SegmentationAnnotation`. (Optional, default value is False)
  * `masks_dir` - path to store segmentation masks in `semantic_only` mode
  * `convert_COCO_to_VOC_labels` - allows to convert COCO labels to Pacsal VOC labels. Optional, default is False.
* `mscoco_mask_rcnn` - converts MS COCO dataset to `ContainerAnnotation` with `DetectionAnnotation` and `CocoInstanceSegmentationAnnotation` named `detection_annotation` and `segmentation_annotation` respectively.
  * `annotation_file` - path to annotation file in json format.
  * `has_background` - allows convert dataset with/without adding background_label. Accepted values are True or False. (default is False).
  * `use_full_label_map` - allows to use original label map (with 91 object categories) from paper instead public available(80 categories).
  * `sort_annotations` - allows to save annotations in a specific order: ascending order of image id or ascending order of image size.
  * `sort_key` - key by which annotations will be sorted (supported keys are `image_id` and `image_size`, default is `image_id`).
  * `dataset_meta_file` - path to json file with dataset meta (e.g. label_map, color_encoding).Optional, more details in [Customizing dataset meta](#customizing-dataset-meta) section.
  * `convert_COCO_to_VOC_labels` - allows to convert COCO labels to Pacsal VOC labels. Optional, default is False.
* `mscoco_keypoints` - converts MS COCO dataset for keypoints localization task to `PoseEstimationAnnotation`.
  * `annotation_file` - path to annotation file in json format.
  * `sort_annotations` - allows to save annotations in a specific order: ascending order of image id or ascending order of image size.
  * `sort_key` - key by which annotations will be sorted (supported keys are `image_id` and `image_size`, default is `image_id`).
  * `remove_empty_images` - allows excluding/inclusing images without objects from/to the dataset..
  * `dataset_meta_file` - path to json file with dataset meta (e.g. label_map, color_encoding).Optional, more details in [Customizing dataset meta](#customizing-dataset-meta) section.
* `wider` - converts from Wider Face dataset to `DetectionAnnotation`.
  * `annotation_file` - path to txt file, which contains ground truth data in WiderFace dataset format.
  * `label_start` - specifies face label index in label map. Default value is 1. You can provide another value, if you want to use this dataset for separate label validation,
  in case when your network predicts other class for faces.
  * `dataset_meta_file` - path to json file with dataset meta (e.g. label_map, color_encoding).Optional, more details in [Customizing dataset meta](#customizing-dataset-meta) section.
* `detection_opencv_storage` - converts detection annotation stored in Detection OpenCV storage format to `DetectionAnnotation`.
  * `annotation_file` - path to annotation in xml format.
  * `image_names_file` - path to txt file, which contains image name list for dataset.
  * `label_start` - specifies label index start in label map. Default value is 1. You can provide another value, if you want to use this dataset for separate label validation.
  * `background_label` - specifies which index will be used for background label. You can not provide this parameter if your dataset has not background label.
  * `dataset_meta_file` - path to json file with dataset meta (e.g. label_map, color_encoding).Optional, more details in [Customizing dataset meta](#customizing-dataset-meta) section.
* `cityscapes` - converts CityScapes Dataset to `SegmentationAnnotation`.
  * `dataset_root_dir` - path to dataset root.
  * `images_subfolder` - path from dataset root to directory with validation images (Optional, default `imgsFine/leftImg8bit/val`).
  * `masks_subfolder` - path from dataset root to directory with ground truth masks (Optional, `gtFine/val`).
  * `masks_suffix` - suffix for mask file names (Optional, default `_gtFine_labelTrainIds`).
  * `images_suffix` - suffix for image file names (Optional, default `_leftImg8bit`).
  * `use_full_label_map` - allows to use full label map with 33 classes instead train label map with 18 classes (Optional, default `False`).
  * `dataset_meta_file` - path to json file with dataset meta (e.g. label_map, color_encoding).Optional, more details in [Customizing dataset meta](#customizing-dataset-meta) section.
* `mapillary_20` - converts Mapillary dataset contained 20 classes to `SegmentationAnnotation`.
  * `data_dir` - path to dataset root folder. Relative paths to images and masks directory determine as `imgs` and `masks` respectively. In way when images and masks are located in non default directories, you can use parameters described below.
  * `images_dir` - path to images folder.
  * `mask_dir` - path to ground truth mask folder.
  * `images_subfolder` - sub-directory for images(Optional, default `imgs`)
  * `mask_subfolder` - sub-directory for ground truth mask(Optional, default `masks`)
* `mapillary_vistas` - converts Mapillary Vistas dataset contained 20 classes to `SegmentationAnnotation`.
  * `data_dir` - path to dataset root folder. Relative paths to images and masks directory determine as `images` and `labels` respectively. In way when images and masks are located in non default directories, you can use parameters described below.
  * `images_dir` - path to images folder.
  * `mask_dir` - path to ground truth mask folder.
  * `images_subfolder` - sub-directory for images(Optional, default `images`)
  * `mask_subfolder` - sub-directory for ground truth mask(Optional, default `labels`)
* `vgg_face` - converts VGG Face 2 dataset for facial landmarks regression task to `FacialLandmarksAnnotation`.
  * `landmarks_csv_file` - path to csv file with coordinates of landmarks points.
  * `bbox_csv_file` - path to cvs file which contains bounding box coordinates for faces (optional parameter).
* `lfw` - converts Labeled Faces in the Wild dataset for face reidentification to `ReidentificationClassificationAnnotation`.
  * `pairs_file` - path to file with annotation positive and negative pairs.
  * `train_file` - path to file with annotation positive and negative pairs used for network train (optional parameter).
  * `landmarks_file` - path to file with facial landmarks coordinates for annotation images (optional parameter).
  * `extension` - images extension(optional, default - `jpg`).
* `face_recognition_bin` - converts preprocessed face recognition dataset stored in binary format to `ReidentificationClassificationAnnotation`.
  * `bin_file` - file with dataset. Example of datasets can be found [here](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo).
  * `images_dir` - directory for saving converted images (Optional, used only if `convert_images` enabled, if not provided `<dataset_root>/converted_images` will be used)
  * `convert_images` - allows decode and save images.
* `mars` - converts MARS person reidentification dataset to `ReidentificationAnnotation`.
  * `data_dir` - path to data directory, where gallery (`bbox_test`) and `query` subdirectories are located.
* `market1501_reid` - converts Market1501 person reidentification dataset to `ReidentificationAnnotation`.
  * `data_dir` - path to data directory, where gallery (`bounding_box_test`) and `query` subdirectories are located.
* `veri776_reid` - converts VeRi776 vehicle reidentification dataset to `ReidentificationAnnotation`.
  * `data_dir` - path to data directory, where gallery (`image_test`) and `image_query` subdirectories are located.
* `image_processing` - converts dataset for common single image processing tasks (e.g. image denoising, style transferring) to `ImageProcessingAnnotation`. This converter is suitable for tasks where model output produced on specific input image should be compared with target image.
  * `data_dir` - path to folder, where images in low and high resolution are located.
  * `input_suffix` - input file name's suffix (default `in`).
  * `target_suffix` - target ground truth file name's suffix (default `out`).
  * `recursive` - enables acquiring of dataset files from `data_dir` subcatalogs (default False).
  * `annotation_loader` - which library will be used for ground truth image reading. Supported: `opencv`, `pillow` (Optional. Default value is pillow). Note, color space of image depends on loader (OpenCV uses BGR, Pillow uses RGB for image reading).
* `parametric_image_processing` - converts dataset for image processing which required variable conditions for getting result, to `ImageProcessingAnnotation. Parameters provided as float value in reference image name using `_` as delimeter.
  * `input_dir` - directory with input images.
  * `reference_dir` - directory with reference images.
  * `annotation_loader` - which library will be used for ground truth image reading. Supported: `opencv`, `pillow` (Optional. Default value is pillow). Note, color space of image depends on loader (OpenCV uses BGR, Pillow uses RGB for image reading).
  * `param_scale` - multiplayer for parameters (Optional, default `0.001`).
* `super_resolution` - converts dataset for single image super resolution task to `SuperResolutionAnnotation`.
  * `data_dir` - path to folder, where images in low and high resolution are located.
  * `lr_dir` - path to directory, where images in low resolution are located.
  * `hr_dir` - path to directory, where images in high resolution are located. **Note:** inside converted annotation, path to directory is not stored, only file name, please use `additional_data_source` for providing prefix.
  * `upsampled_dir` - path to directory, where upsampled images are located, if 2 streams used.
  * `ignore_suffixes` - matched low resolution, high resolution image located in different directories without usage suffixes, using numeric ids (Optional, default False).
  * `lr_suffix` - low resolution file name's suffix (default lr).
  * `hr_suffix` - high resolution file name's suffix (default hr).
  * `annotation_loader` - which library will be used for ground truth image reading. Supported: `opencv`, `pillow`, `pillow_rgb` (for explicit data conversion to RGB format), `dicom`, `skimage`. (Optional. Default value is pillow). Note, color space of image depends on loader (OpenCV uses BGR, Pillow uses RGB for image reading).
  * `two_streams` - enable 2 input streams where usually first for original image and second for upsampled image. (Optional, default False).
  * `upsample_suffix` - upsample images file name's suffix (default upsample).
* `super_resolution_dir_based` - converts dataset for single image super resolution task to `SuperResolutionAnnotation` which have directory based structure (high resolution and low resolution images located on separated directories and matches by name or numeric id).
The main difference between this converter and `super_resolution` in data organization. `super_resolution` converter should be used if all high and low resolution images located in the same dir and have difference in suffixes.
  * `annotation_loader` - which library will be used for ground truth image reading. Supported: `opencv`, `pillow`, `pillow_rgb` (for explicit data conversion to RGB format), `dicom`, `skimage`. (Optional. Default value is pillow). Note, color space of image depends on loader (OpenCV uses BGR, Pillow uses RGB for image reading).
  * `two_streams` - enable 2 input streams where usually first for original image and second for upsampled image. (Optional, default False).
  * `images_dir` - path to dataset root, where directories with low and high resolutions are located.
  * `lr_dir` - path to directory, where images in low resolution are located (Optional, default `<images_dir>/LR`).
  * `hr_dir` - path to directory, where images in high resolution are located (Optional, default `<images_dir>/HR`). **Note:** inside converted annotation, path to directory is not stored, only file name, please use `additional_data_source` for providing prefix.
  * `upsampled_dir` - path to directory, where upsampled images are located, if 2 streams used (Optional, default `<images_dir>/upsample`).
  * `relaxed_names` - allow to use more relaxed search of high resolution or/and upsampled images matching only numeric ids. Optional, by default full name matching required.
  * `hr_prefixed` - allow to use partial name matching  when low resolution filename is a part of high resolution filename. Not applicable when `relaxed_names` is set. Optional, by default full name matching required.
* `multi_frame_super_resolution` - converts dataset for super resolution task with multiple input frames usage.
    * `data_dir` - path to folder, where images in low and high resolution are located.
    * `lr_suffix` - low resolution file name's suffix (default lr).
    * `hr_suffix` - high resolution file name's suffix (default hr).
    * `annotation_loader` - which library will be used for ground truth image reading. Supported: `opencv`, `pillow` (Optional. Default value is pillow). Note, color space of image depends on loader (OpenCV uses BGR, Pillow uses RGB for image reading).
    * `number_input_frames` - the number of input frames per inference.
    * `reference_frame` - the id of frame in sample frame sequence used for matching with high resolution. You can define number of frame or choose one of predefined: `first` (first frame used as reference), `middle` (`num_frames` / 2), `last` (last frame in sequence).
* `multi_target_super_resolution` - converts dataset for single image super resolution task with multiple target resolutions to `ContainerAnnotation` with `SuperResolutionAnnotation` representations for each target resolution.
   * `data_dir` - path to dataset root, where directories with low and high resolutions are located.
   * `lr_path` - path to low resolution images directory relative to `data_dir`.
   * `hr_mapping` - dictionary which represent mapping between target resolution and directory with images. Keys are also used as keys for `ContainerAnnotation`. All paths should be relative to `data_dir`.
* `icdar_detection` - converts ICDAR13 and ICDAR15 datasets for text detection challenge to `TextDetectionAnnotation`.
  * `data_dir` - path to folder with annotations on txt format.
  * `word_spotting` - if it is true then transcriptions that have lengths less than 3 symbols or transcriptions containing non-alphanumeric symbols will be marked as difficult.
* `icdar13_recognition` - converts ICDAR13 dataset for text recognition task to `CharacterRecognitionAnnotation`.
  * `annotation_file` - path to annotation file in txt format.
  * `delimeter` - delimeter between image and text for recognition. Supported values - `space` and `tab` for space and tabular separator respectively.
* `lmdb_text_recognition_database` - converter for text recognition dataset in a form of LMDB database.
  * `lower_case` - parameter describing if ground truth text should be converted to lower case.
* `unicode_character_recognition` - converts [Kondate](http://web.tuat.ac.jp/~nakagawa/database/en/kondate_about.html) dataset and [Nakayosi](http://web.tuat.ac.jp/~nakagawa/database/en/about_nakayosi.html) for handwritten Japanese text recognition task , and [SCUT-EPT](https://github.com/HCIILAB/SCUT-EPT_Dataset_Release) for handwritten simplified Chinese text recognition task to `CharacterRecognitionAnnotation`.
  * `annotation_file` - path to annotation file in txt format.
  * `decoding_char_file` - path to decoding_char_file, consisting of all supported characters separated by '\n' in txt format.
* `brats` - converts BraTS dataset format to `BrainTumorSegmentationAnnotation` format. Also can be used to convert other nifti-based datasets.
  * `data_dir` - dataset root directory, which contain subdirectories with validation data (`imagesTr`) and ground truth labels (`labelsTr`).
  Optionally you can provide relative path for these subdirectories (if they have different location) using `image_folder` and `mask_folder` parameters respectively.
  * `mask_channels_first` - allows read gt mask nifti files and transpose in order where channels first (Optional, default `False`)
  * `labels_file` - path to file, which contains labels (optional, if omitted no labels will be shown)
  * `relaxed_names` - allows to use more relaxed search of labels matching only numeric ids. Optional, by default full name matching required.
  * `multi_frame` - allows to convert annotation of 3D images as sequence of 2D frames (optional, default `False`)
  * `frame_separator` - string separator between file name and frame number in `multi_frame` (optional, default `#`)
  * `frame_axis` - number of frame axis in 3D Image (optional, default `-1`, last axis)
  * `as_regression` - allows dataset conversion as `NiftiRegressionAnnotation` annotation (optional, default `False`)
* `k_space_mri` - converts `k-spaced MRI` dataset format to `ImageRepresentationAnnotation` format. MRI datasets, for example `Calgary-Campinas`, provides data in Fourier images form (so called k-space images). Converter performs dataset annotation and preprocessing of ground truth images and model input.
  * `data_dir` - path to dataset root
  * `image_folder`- path to source k-space files directory, relatively `data_dir` (optional, default `images`)
  * `reconstructed_folder`- path to reconstructed images directory, relatively `data_dir` (optional, default `reconstructed`)
  * `masked_folder`- path to masked k-space files directory, relatively `data_dir` (optional, default `masked`)
  * `mask_file` - k-space mask filename
  * `stats_file` - k-space normalization factors filename
  * `skip_dumps` - allows dataset annotation without preprocessing
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
  * `dataset_meta_file` - path to json file with prepared dataset meta info. It should contain `label_map` key with dictionary in format class_id: class_name and optionally `segmentation_colors` (if your dataset uses color encoding). Segmentation colors is a list of channel-wise values for each class. (e.g. if your dataset has 3 classes in BGR colors, segmentation colors for it will looks like: `[[255, 0, 0], [0, 255, 0], [0, 0, 255]]`). (Optional, you can provide self-created file as `dataset_meta` in your config).
**Note: since OpenVINO 2020.4 converter behaviour changed. `data_source` parameter of dataset should contain the directory for images only, if you have segmentation mask in separated location, please use `segmentation_masks_source` for specifying gt masks location.**
* `background_matting` - converts general format of datasets for background matting task to `BackgroundMattingAnnotation`. The converter expects following dataset structure:
  1. images and GT masks are located in separated directories (e.g. `<dataset_root>/images` for images and `<dataset_root>/masks` for masks respectively)
  2. images and GT masks has common part in names and can have difference in prefix and postfix (e.g. image name is image0001.jpeg, mask for it is gt0001.png are acceptable. In this case base_part - 0001, image_prefix - image, image_postfix - .jpeg, mask_prefix - gt, mask_postfix - .png)
  * `images_dir` - path to directory with images.
  * `masks_dir` - path to directory with GT masks.
  * `image_prefix` - prefix part for image file names. (Optional, default is empty).
  * `image_postfix` - postfix part for image file names (optional, default is `.png`).
  * `mask_prefix` - prefix part for mask file names. (Optional, default is empty).
  * `image_postfix` - postfix part for mask file names (optional, default is `.png`).
* `camvid` - converts CamVid dataset with 12 classes to `SegmentationAnnotation`. Dataset can be found in the following [repository](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid)
  * `annotation_file` - file in txt format which contains list of validation pairs (`<path_to_image>` `<path_to_annotation>` separated by space)
  * `dataset_meta_file` - path to json file with dataset meta (e.g. label_map, color_encoding).Optional, more details in [Customizing dataset meta](#customizing-dataset-meta) section.
* `camvid_32` - converts CamVid dataset with 32 classes to `SegmentationAnnotation`. Dataset can be found [here](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/).
  * `labels_dir` - directory with labeled ground truth images.
  * `images_dir` - directory with input data.
  * `val_subset_ratio` - ratio of subset, which should be used for validation. It is the float value in (0, 1] range for definition subset size as `<total_dataset_size> * <subset_ratio>`. Optional, default 1 (it means full dataset used for validation).
  * `dataset_meta_file` - path to json file with dataset meta (e.g. label_map, color_encoding).Optional, more details in [Customizing dataset meta](#customizing-dataset-meta) section.
* `image_retrieval` - converts dataset for image retrieval task to `ReidentificationAnnotation`. Dataset should have following structure:
   1. the dataset root directory contains 2 subdirectory named `gallery` and `queries` for gallery images and query images respectively.
   2. Every of these subdirectories should contains text file with list of pairs: `<path_to_image>` `<image_ID>` (image_path and image_ID should be separated by space),  where `<path_to_image>` is path to the image related dataset root, `<image_ID>` is the number which represent image id in the gallery.
   * `data_dir` - path to dataset root directory.
   * `gallery_annotation_file` - file with gallery images and IDs concordance in txt format (Optional, default value is `<data_dir>/gallery/list.txt`)
   * `queries_annotation_file` - file with queries images and IDs concordance in txt format (Optional, default value is `<data_dir>/queries/list.txt`)
* `cvat_object_detection` - converts [CVAT XML annotation version 1.1](https://openvinotoolkit.github.io/cvat/docs/for-developers/xml_format/) format for images to `DetectionAnnotation`.
  * `annotation_file` - path to xml file in appropriate format.
  * `has_background` - allows prepend original labels with special class represented background and convert dataset for n+1 classes instead n (default value is True).
  * `dataset_meta_file` - path to json file with dataset meta (e.g. label_map, color_encoding).Optional, more details in [Customizing dataset meta](#customizing-dataset-meta) section.
* `cvat_attributes_recognition` - converts [CVAT XML annotation version 1.1](https://openvinotoolkit.github.io/cvat/docs/for-developers/xml_format/) format for images to `ClassificationAnnotation` or `ContainerAnnotation` with `ClassificationAnnotation` as value type and attribute names as keys (in multiple attributes case). Used bbox attributes as annotation classes.
  * `annotation_file` - path to xml file in appropriate format.
  * `label` - the dataset label which will be used for attributes collection (e.g. if your dataset contains 2 labels: `face` and `person` and you want recognise attributes for face, you should use `face` as value for this parameter).
* `cvat_age_gender` -  converts [CVAT XML annotation version 1.1](https://openvinotoolkit.github.io/cvat/docs/for-developers/xml_format/) format for images which represent dataset for age gender recognition to `ContainerAnnotation` with `ClassificationAnnotation` for gender recognition, `ClassificationAnnotation` for age classification and `RegeressionAnnotation` for age regression. The identifiers for representations following: `gender_annotation`, `age_class_annotation`, `age_regression_annotation`.
  * `annotation_file` - path to xml file in appropriate format.
* `cvat_facial_landmarks` - converts [CVAT XML annotation version 1.1](https://openvinotoolkit.github.io/cvat/docs/for-developers/xml_format/) format for images to `FacialLandmarksAnnotation`.
  * `annotation_file` - path to xml file in appropriate format.
* `cvat_pose_estimation` - converts [CVAT XML annotation version 1.1](https://openvinotoolkit.github.io/cvat/docs/for-developers/xml_format/) format for images to `PoseEstimationAnnotation`.
  * `annotation_file` - path to xml file in appropriate format.
* `cvat_text_recognition` - converts [CVAT XML annotation version 1.1](https://openvinotoolkit.github.io/cvat/docs/for-developers/xml_format/) format for images to `CharacterRecognitionAnnotation`.
  * `annotation_file` - path to xml file in appropriate format.
* `cvat_binary_multilabel_attributes_recognition` - converts [CVAT XML annotation version 1.1](https://openvinotoolkit.github.io/cvat/docs/for-developers/xml_format/) format for images to `MultiLabelRecognitionAnnotation`. Used bbox attributes as annotation classes. Each attribute field should contains `T` or `F` values for attribute existence/non-existence on the image respectively.
  * `annotation_file` - path to xml file in appropriate format.
  * `label` - the dataset label which will be used for attributes collection (e.g. if your dataset contains 2 labels: `face` and `person` and you want recognise attributes for face, you should use `face` as value for this parameter).
* `cvat_person_detection_action_recognition` converts dataset with [CVAT XML annotation version 1.1](https://openvinotoolkit.github.io/cvat/docs/for-developers/xml_format/) for person detection and action recognition task to `ContainerAnnotation` with `DetectionAnnotation` for person detection quality estimation named `person_annotation` and `ActionDetectionAnnotation` for action recognition named `action_annotation`.
  * `annotation_file` - path to xml file with ground truth.
  * `use_case` - use case, which determines the dataset label map. Supported range actions:
    * `common_3_actions`(seating, standing, raising hand)
    * `common_6_actions`(seating, writing, raising hand, standing, turned around, lie on the desk)
    * `teacher` (standing, writing, demonstrating)
    * `raising_hand` (seating, raising hand)
* `lpr_txt` - converts annotation for license plate recognition task in txt format to `CharacterRecognitionAnnotation`.
  * `annotation_file` - path to txt annotation.
  * `decoding_dictionary` - path to file containing dictionary for output decoding.
* `squad_emb` - converts the Stanford Question Answering Dataset ([SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)) to `Question Answering Embedding Annotation`. **Note: This converter not only converts data to metric specific format but also tokenize and encodes input for BERT.**
  * `testing_file` - path to testing file.
  * `vocab_file` - path to model co vocabulary file.
  * `max_seq_length` - maximum total input sequence length after word-piece tokenization (Optional, default value is 128).
  * `max_query_length` - maximum number of tokens for the question (Optional, default value is 64).
  * `lower_case` - allows switching tokens to lower case register. It is useful for working with uncased models (Optional, default value is False)
* `squad` - converts the Stanford Question Answering Dataset ([SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)) to `Question Answering Annotation`. **Note: This converter not only converts data to metric specific format but also tokenize and encodes input for BERT.**
  * `testing_file` - path to testing file.
  * `vocab_file` - path to model co vocabulary file.
  * `max_seq_length` - maximum total input sequence length after word-piece tokenization (Optional, default value is 128).
  * `max_query_length` - maximum number of tokens for the question (Optional, default value is 64).
  * `doc_stride` -stride size between chunks for splitting up long document (Optional, default value is 128).
  * `lower_case` - allows switching tokens to lower case register. It is useful for working with uncased models (Optional, default value is False)
* `squad_bidaf` - converts the Stanford Question Answering Dataset ([SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)) to `QuestionAnsweringBiDAFAnnotation`. **Note:** This converter not only converts data to metric specific format but also tokenize and encodes input for BiDAF using nltk.word_tokenize.
  * `testing_file` - path to testing file.
* `xnli` - converts The Cross-lingual Natural Language Inference Corpus ([XNLI](https://github.com/facebookresearch/XNLI)) to `TextClassificationAnnotattion`. **Note: This converter not only converts data to metric specific format but also tokenize and encodes input for BERT.**
  * `annotation_file` - path to dataset annotation file in tsv format.
  * `vocab_file` -  path to model vocabulary file for WordPiece tokinezation (Optional in case, when another tokenization approach used).
  * `sentence_piece_model_file` - model used for [SentencePiece](https://github.com/google/sentencepiece) tokenization (Optional in case, when another tokenization approach used).
  * `max_seq_length` - maximum total input sequence length after word-piece tokenization (Optional, default value is 128).
  * `lower_case` - allows switching tokens to lower case register. It is useful for working with uncased models (Optional, default value is False).
  * `language_filter` - comma-separated list of used in annotation language tags for selecting records for specific languages only. (Optional, if not used full annotation will be converted).
* `mnli` - converts The Multi-Genre Natural Language Inference Corpus ([MNLI](https://cims.nyu.edu/~sbowman/multinli/)) to `TextClassificationAnnotattion`. **Note: This converter not only converts data to metric specific format but also tokenize and encodes input for BERT.**
  * `annotation_file` - path to dataset annotation file in tsv format.
  * `vocab_file` - path to model vocabulary file for WordPiece tokinezation. (Optional, can be not provided in case, when another tokenization approach used.)
  * `sentence_piece_model_file` - model used for [SentencePiece](https://github.com/google/sentencepiece) tokenization (Optional in case, when another tokenization approach used).
  * `max_seq_length` - maximum total input sequence length after tokenization (Optional, default value is 128).
  * `lower_case` - allows switching tokens to lower case register. It is useful for working with uncased models (Optional, default value is False).
* `mrpc` - converts The Microsoft Research Paraphrase Corpus ([MRPC](https://www.microsoft.com/en-us/download/details.aspx?id=52398)) to `TextClassificationAnnotattion`. **Note: This converter not only converts data to metric specific format but also tokenize and encodes input for BERT.**
  * `annotation_file` - path to dataset annotation file in tsv format.
  * `vocab_file` - path to model vocabulary file for WordPiece tokenization. (Optional, can be not provided in case, when another tokenization approach used.)
  * `sentence_piece_model_file` - model used for [SentencePiece](https://github.com/google/sentencepiece) tokenization (Optional in case, when another tokenization approach used).
  * `max_seq_length` - maximum total input sequence length after tokenization (Optional, default value is 128).
  * `lower_case` - allows switching tokens to lower case register. It is useful for working with uncased models (Optional, default value is False).
* `cola` - converts The Corpus of Linguistic Acceptability ([CoLA](https://nyu-mll.github.io/CoLA/)) to `TextClassificationAnnotattion`. **Note: This converter not only converts data to metric specific format but also tokenize and encodes input for BERT.**
  * `annotation_file` - path to dataset annotation file in tsv format.
  * `vocab_file` - path to model vocabulary file for WordPiece tokinezation. (Optional, can be not provided in case, when another tokenization approach used.)
  * `sentence_piece_model_file` - model used for [SentencePiece](https://github.com/google/sentencepiece) tokenization (Optional in case, when another tokenization approach used).
  * `max_seq_length` - maximum total input sequence length after tokenization (Optional, default value is 128).
  * `lower_case` - allows switching tokens to lower case register. It is useful for working with uncased models (Optional, default value is False).
* `cola` - converts The Corpus of Linguistic Acceptability ([CoLA](https://nyu-mll.github.io/CoLA/)) to `TextClassificationAnnotattion`. **Note: This converter not only converts data to metric specific format but also tokenize and encodes input for BERT.**
  * `annotation_file` - path to dataset annotation file in tsv format.
  * `vocab_file` - path to model vocabulary file for WordPiece tokenization. (Optional, can be not provided in case, when another tokenization approach used.)
  * `sentence_piece_model_file` - model used for [SentencePiece](https://github.com/google/sentencepiece) tokenization (Optional in case, when another tokenization approach used).
  * `max_seq_length` - maximum total input sequence length after tokenization (Optional, default value is 128).
  * `lower_case` - allows switching tokens to lower case register. It is useful for working with uncased models (Optional, default value is False).
* `imdb` - converts [IMDB sentiment dataset](https://ai.stanford.edu/~amaas/data/sentiment/) to `TextClassificationAnnotattion`. **Note: This converter not only converts data to metric specific format but also tokenize and encodes input for BERT.**
  * `annotation_file` - path to dataset annotation file in tsv format.
  * `vocab_file` - path to model vocabulary file for WordPiece tokinezation. (Optional, can be not provided in case, when another tokenization approach used.)
  * `sentence_piece_model_file` - model used for [SentencePiece](https://github.com/google/sentencepiece) tokenization (Optional in case, when another tokenization approach used).
  * `max_seq_length` - maximum total input sequence length after tokenization (Optional, default value is 128).
  * `lower_case` - allows switching tokens to lower case register. It is useful for working with uncased models (Optional, default value is False).
* `bert_xnli_tf_record` - converts The Cross-lingual Natural Language Inference Corpus ([XNLI](https://github.com/facebookresearch/XNLI)) stored in tf records format. This converter usage requires TensorFlow installation. Please make sure that TensorFlow installed before conversion.
  * `annotattion_file` - path to annotation file in tf records format.
* `cmu_panoptic_keypoints` - converts CMU Panoptic dataset to `PoseEstimation3dAnnotation` format.
  * `data_dir` - dataset root directory, which contain subdirectories with validation scenes data.
* `clip_action_recognition` - converts annotation video-based action recognition datasets. Before conversion validation set should be preprocessed using approach described [here](https://github.com/openvinotoolkit/training_extensions/blob/develop/misc/pytorch_toolkit/action_recognition/README.md#preparation).
  * `annotation_file` - path to annotation file in json format.
  * `data_dir` - path to directory with prepared data (e. g. data/kinetics/frames_data).
  * `clips_per_video` - number of clips per video (Optional, default 3).
  * `clip_duration` - clip duration (Optional, default 16)
  * `temporal_stride` - temporal stride for frames selection (Optional, default 2).
  * `numpy_input` - allows usage numpy files instead images. It can be useful if data required difficult preprocessing steps (e.g. conversion to optical flow) (Optional, default `False`)
  * `subset` - dataset split: `train`, `validation` or `test` (Optional, default `validation`).
  * `dataset_meta_file` - path to json file with dataset meta (e.g. label_map, color_encoding).Optional, more details in [Customizing dataset meta](#customizing-dataset-meta) section.
  * `num_samples` - select first n samples from dataset (Optional, if not provided full subset of samples will be used).
* `continuous_clip_action_recognition` - converts annotation of video-based MS-ASL dataset to `ClassificationAnnotation`.
  * `annotation_file` - path to annotation file in txt format.
  * `data_dir` - dataset root directory, which contains subdirectories with extracted video frames.
  * `out_fps` - output frame rate of generated video clips.
  * `clip_length` - number of frames of generated video clips.
  * `img_prefix` - prefix for used images. (Optional, default - `img_`).
* `redweb` - converts [ReDWeb](https://sites.google.com/site/redwebcvpr18) dataset for monocular relative depth perception to `DepthEstimationAnnotation`
  * `data_dir` - the dataset root directory, where `imgs` - directory with RGB images and `RD` - directory with relative depth maps are located (Optional, if you want to provide `annotation_file`)
  * `annotation_file`- the file in txt format which contains pairs of image and depth map files. (Optional, if not provided full content of `data_dir` will be considered as dataset.)
* `nyu_depth_v2` - converts [NYU Depth Dataset V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) for depth estimation to `DepthEstimationAnnotation`. This converter accept preprocessed data stored in HDF5 format, which can be downloaded from this [page](http://datasets.lids.mit.edu/fastdepth/data/)
  * `data_dir` - directory with HDF5 files. (Optional, can be omitted if you already have converted images and depth maps).
  * `images_dir` - directory for images. If `allow_convert_data` is True, the directory will be used for saving converted images, otherwise used for data reading. (Optional, can be not provided in conversion case, default value `<data_dir>/converted/images`).
  * `depth_map_dir` - directory for reference depth maps, stored in numpy format. If `allow_convert_data` is True, the directory will be used for saving converted depth maps, otherwise used for data reading.
    (Optional, can be not provided in conversion case, default value `<data_dir>/converted/depth`). Please, note, you need to specify path to directory with depth maps with `additional_data_source` parameter in your config during evaluation.
  * `allow_convert_data` - allows to convert data from HDF5 format (Optional, default False).
* `inpainting` - converts images to `ImageInpaintingAnnotation`.
  * `images_dir` - path to images directory.
  * `masks_dir` - path to mask dataset to be used for inpainting (Optional).
* `aflw2000_3d` - converts [AFLW2000-3D](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm) dataset for 3d facial landmarks regression task to `FacialLandmarks3DAnnotation`.
   * `data_dir` - directory, where input images and annotation files in MATLAB format stored.
* `style_transfer` - converts images to `StyleTransferAnnotation`.
  * `images_dir` - path to images directory.
* `ade20k` - converts ADE20K dataset to `SegmentationAnnotation`.
  * `images_dir` - path to directory with images (e.g. `ADEChallengeData2016/images/validation`).
  * `annotations_dir` - path to directory with annotations (e.g. `ADEChallengeData2016/annotations/validation`).
  * `object_categories_file` - path to file with labels (e.g. `ADEChallengeData2016/objectInfo150.txt`).
  * `num_classes` - number of used classes.
* `criteo_kaggle_dac` - converts Criteo datasets to `ClassificationAnnotation`.
  * `testing_file` - path to preprocessed Criteo file (e.g. `criteo/terabyte/terabyte_preprocessed,npz`).
  * `batch` - batch size expected by model
  * `subsample_size` - number of batches in test-only dataset, If provided, total number of records is batch * subsample_size
  * `validation` - if provided, only second half of dataset converted to annotations, according to dataset definition
  * `preprocessed_dir` - path to store preprocessed batch files (e.g. `criteo/terabyte/preprocessed`).
  * `separator` - symbol used to separate feature identifiers from batch data filename.
  * `save_preprocessed_features` - allow to save preprocessed input features into `preprocessed_dir` (Optional, default True).
* `features_regression` - converts dataset stored in format of directories with preprocessed input numeric data (features) in text files and reference data in the same format to `FeatureRegressionAnnotation`.
 This approach allows comparing output of model from different frameworks (e.g. OpenVINO converted model and source framework realisation).
  * `input_dir` - directory with input data files.
  * `reference_dir` - directory with reference data. **Note: inside converted annotation, path to directory is not stored, only file name, please use `additional_data_source` for providing prefix.**
  * `input_suffix` - suffix for input files (usually file extension). Optional, default `.txt`.
  * `reference_suffix` - suffix for reference files (usually file extension). Optional, default `.txt`.
  * `use_bin_data` - this flag specifies that input data in binary format, optional, default `False`
  * `bin_data_dtype` - data type for reading binary data.
* `multi_feature_regression` - converts dataset stored in format of directories with preprocessed input numeric data (features) in dictionary format, where keys are layer names and values - features and reference data in the same format to `FeatureRegressionAnnotation`.
 This approach allows comparing output of model from different frameworks (e.g. OpenVINO converted model and source framework realisation). Please note, that input and reference should be stored as dict-like objects in npy files.
  * `data_dir` - directory with input and reference files.
   * `input_suffix` - suffix for input files (usually file extension). Optional, default `in.npy`.
   * `reference_suffix` - suffix for reference files (usually file extension). Optional, default `out.npy`.
   * `prefix` - prefix for input files selection (Optional, ignored if not provided).
* `librispeech` - converts [librispeech](http://www.openslr.org/12) dataset to `CharachterRecognitionAnnotation`.
  * `data_dir` - path to dataset directory, which contains converted wav files.
  * `annotation_file` - path to file which describe the data which should be used in evaluation (`audio_filepath`, `text`, `duration`). Optional, used only for data filtering and sorting audio samples by duration.
  * `use_numpy` - allows to use preprocessed data stored in npy-files instead of audio (Optional, default False).
  * `top_n` - numeric value for getting only the n shortest samples **Note:** applicable only with `annotation_file` providing.
  * `max_duration` - maximum clip duration to include into annotation. Default 0, means no duration checking.
* `criteo` - converts [Criteo](http://labs.criteo.com/2013/12/download-terabyte-click-logs/) datasets to `ClassificationAnnotation`.
  * `testing_file` - Path to testing file, terabyte_preprocessed.npz (Criteo Terabyte) or day_6_processed.npz (Criteo Kaggle Dac)
  * `batch` - Model batch.
  * `subsample_size` - Subsample size in batches
  * `validation` - Allows to use half of dataset for validation purposes
  * `block` - Make batch-oriented annotations
  * `separator` - Separator between input identifier and file identifier
  * `preprocessed_dir` - Preprocessed dataset location
  * `dense_features` - Name of model dense features input
  * `sparse_features` - Name of model sparse features input. For multiple inputs use comma-separated list in form `<name>:<index>`
  * `lso_features` - Name of lS_o-like features input
* `im2latex_formula_recognition` - converts im2latex-like datasets to `CharacterRecognitionAnnotation`. [Example of the dataset](http://lstm.seas.harvard.edu/latex/data/)
  * `images_dir` - path to input images (rendered or scanned formulas)
  * `formula_file` - path to file containing one formula per line
  * `split_file` - path to file containing `img_name` and corresponding formula `index` in `formula_file` separated by tab per line
  * `vocab_file` - file containing vocabulary to cast token class indices into human-readable tokens
* `dna_sequence` - converts dataset for DNA sequencing to `DNASequenceAnnotation`.
  * `chunks_file` - npy file with input chunks.
  * `ref_file` - npy file with reference sequence.
  * `num_chunks` - subset size for usage in validation, if not provided the whole dataset will be used.
  * `alphabet` - alphabet for sequence decoding (Optional, default ["N", "A", "C", "G", "T"]).
* `place_recognition` - converts dataset for image based localization task to `PlaceRecognitionAnnotation`
  * `subset_file` - matlab file contains info about subset used in validation.
* `mpii` - converts MPII Human Pose Estimation dataset to `PoseEstimationAnnotation`.
  * `annotation_file` - json-file with annotation.
  * `headboxes_file` - numpy file with boxes contained head coordinates for each image.
  * `gt_pos_file` - numpy file with ground truth keypoints, optional, if not provided, default keypoints from annotation will be used.
  * `joints_visibility_file` - numpy file with ground truth keypoints visibility level, optional, if not provided, default visibility level from annotation will be used.
* `cluttered_mnist` - converts MNIST dataset from spatial transformer network [example](https://github.com/oarriaga/STN.keras/tree/master/datasets) to `ClassificationAnnotation`.
  * `data_file` - npz file with dataset.
  * `split` - dataset split: `train` - for training subset, `valid` - for train-validation subset, `test` - for testing subset (Optional, default test).
  * `convert_images` - allows convert images from raw data stored in npz and save them into provided directory (Optional, default True).
  * `images_dir` - directory for saving converted images (Optional, if not provided, the images will be saved into converted_images directory in the same location, where data_file is stored)
* `antispoofing` - converts dataset for antispoofing classification task to `ClassificationAnnotation`
  * `data_dir` - path to root folder of the dataset
  * `annotation_file` - path to json file containing annotations to the dataset ({index: {path:"...", labels:[...], bbox:[...] (optional), ...})
  * `label_id` - number of label in the annotation file representing spoof/real labels
  * `dataset_meta_file` - path to json file with dataset meta (e.g. label_map)
* `sound_classification` - converts dataset for sound classification to `ClassificationAnnotation`. The dataset should be represented by directory with input wav files and annotation in 2 column csv format, where first column is audio file name and second is label id from dataset.
  * `annotation_file` - csv file with selected subset for evaluation, file structure described above.
  * `audio_dir` - directory with input data, (optional, required only if you want check file existence during annotation conversion).
* `ade20k_image_translation` - converts ADE20K dataset to `ImageProcessingAnnotation` according to `reference_file`.
  * `annotations_dir` - path to directory with annotations (e.g. `ADEChallengeData2016/annotations`).
  * `reference_file` - path to file with pairs key (validation): value (train).
* `salient_object_detection` - converts dataset for salient object detection to `SalientRegionAnnotation`. The dataset should have following structure:
  1. images have numeric ids like names and `jpg` extension (e.g. image/0.jpg, image/1.jpg, image/2.jpg, ...).
  2. salience map located in separated directory, have the same ids like images and `png` extension  (e.g. mask/0.png, mask/1.png, mask/2.png).
  * `images_dir` - directory with input images.
  * `masks_dir` - directory with reference salience maps.
  * `annotation_file` - txt file with selected image ids.
* `wflw` - converts WFLW dataset for facial landmarks regression task to `FacialLandmarksAnnotation`.
  * `annotation_file` - path to txt file with ground truth data in WFLW dataset format.
  * `images_dir` - path to dataset images, used only for content existence check (optional parameter).
* `common_object_detection` - converts object detection dataset to `DetectionAnnotation`. Dataset should be stored in following format:
  1. labels_map defined as text file, where defined labels line by line.
  2. annotations for each image stored in separated text file. Box is represented by space separated info: <label_id> <x_min> <y_min> <x_max> <y_max>.
  3. name of annotation file the same like image name (or additional file with file mapping should be defined).
  * `annotation_dir` - path to directory with annotation files.
  * `images_dir` - path to directory with images (Optional, used only for content check step).
  * `labels_file` - path to file with labels.
  * `pairs_file` - path to file where described image and annotation file pairs (Optional, if not provided list will be created according to annotation_dir content).
  * `has_background` - flag that background label should be added to label_map (Optional, default False).
  * `add_background_to_label_id` - flag that label_ids defined in annotation should be shifted if `has_background` enabled.
* `see_in_the_dark` - converts See-in-the-Dark dataset described in the [paper](https://cchen156.github.io/paper/18CVPR_SID.pdf) to `ImageProcessingAnnotation`.
  * `annotation_file` - path to image pairs file in txt format.
* `conll_ner` - converts CONLL 2003 dataset for Named Entity Recognition to `BERTNamedEntityRecognitionAnnotation`.
  * `annotation_file` - annotation file in txt forma
  * `vocab_file` - vocab file for word piece tokenization.
  * `lower_case` - converts all tokens to lower case during tokenization (Optional, default `False`).
  * `max_length` - maximal input sequence length (Optional, default 128).
  * `pad_input` - allow padding for input sequence if input less that `max_length` (Optional, default `True`).
  * `include_special_token_lables` - allow extension original dataset labels with special token labels (`[CLS'`, `[SEP]`]) (Optional, default `False`).
* `tacotron2_data_converter` - converts input data for custom tacotron2 pipeline.
  * `annotation_file` - tsv file with location input data and reference.
* `noise_suppression_dataset` - converts dataset for audio denoising to `NoiseSuppressionAnnotation`
  * `annotation_file` - txt file with file pairs `<clean_signal> <noisy_signal>`.
* `vimeo90k` - converts Vimeo-90K dataset for a systematic evaluation of video processing algorithms to `SuperResolutionAnnotation`.
  * `annotation_file` - path to text file with list of dataset setuplets included in test.
  * `add_flow` - allows annotation of flow data (optional, default `False`).
* `kaldi_asr_data` - converts preprocessed Kaldi\* features dataset to `CharacterRecognitionAnnotation`.
   * `annotation_file` - file with gt transcription table.
   * `data_dir` - directory with ark files.
   * `features_subset_file` - file with list testing ark files, Optional, if not provided, all found in `data_dir` files will be used.
   * `ivectors` - include ivectors features to input, Optional, default `False`.
* `kaldi_feat_regression` - converts preprocessed Kaldi\* features to `RegressionAnnotation`.
  * `data_dir` - directory with input ark files.
  * `features_subset_file` - file with list testing ark files, Optional, if not provided, all found in `data_dir` files will be used.
  * `ivectors` - include ivectors features to input, Optional, default `False`.
  * `ref_data_dir` - directory with reference ark files (Optional, if not provided `data_dir` will be used instead).
  * `vectors_mode` - allow usage each vector in utterance as independent data.
  * `ref_file_suffix` - suffix for search reference files (Optional, default `_kaldi_score`).
* `electricity` - converts Electricity dataset to `TimeSeriesForecastingAnnotation`.
  * `data_path_file` - Path to dataset file in .csv format.
  * `num_encoder_steps` - The maximum number of historical timestamps that model use.
* `yolo_labeling` - converts object detection dataset with annotation in YOLO labeling format to `DetectionAnnotation`.
  * `annotations_dir` - path to directory with annotation files in txt format.
  * `images_dir` -  path to directory with images (Optional).
  * `labels_file` - path to file with labels in txt format (Optional).
  * `images_suffix` - suffix for image file names (Optional, default: `.jpg`).
* `label_me_detection` - converts dataset obtained using [LabelMe](http://labelme.csail.mit.edu/Release3.0/) Annotation Tool to `DetectionAnnotation`.
  * `annotations_dir` - path to directory with annotation files in xml format.
  * `dataset_meta_file` - path to json file with dataset meta (e.g. label_map, color_encoding). More details in [Customizing dataset meta](#customizing-dataset-meta) section.
  * `images_dir` - path to directory with images (Optional).
  * `has_background` - allows convert dataset with/without adding background_label (Optional, default: False).
* `label_me_segmentation` - converts dataset obtained using [LabelMe](http://labelme.csail.mit.edu/Release3.0/) Annotation Tool to `SegmentationAnnotation`.
  * `annotations_dir` - path to directory with annotation files in xml format.
  * `dataset_meta_file` - path to json file with dataset meta (e.g. label_map, color_encoding). More details in [Customizing dataset meta](#customizing-dataset-meta) section.
  * `images_dir` - path to directory with images (Optional).
  * `masks_dir` - path to directory with ground truth segmentation masks (Optional).
* `cls_dataset_folder` - converts generic classification dataset with [DatasetFolder](https://pytorch.org/vision/stable/datasets.html#datasetfolder) format to `ClassificationAnnotation`.
  * `data_dir` - directory with input images in following structure:
    ```
        data_dir/class_a/xxx.ext
        data_dir/class_a/xxy.ext
        data_dir/class_b/[...]/xxz.ext
        ...
        data_dir/class_y/123.ext
        data_dir/class_z/nsdf3.ext
        data_dir/class_z/[...]/asd932_.ext
    ```
* `open_images_detection` - converts Open Images dataset for object detection task to `DetectionAnnotation`.
  * `bbox_csv_file` - path to cvs file which contains bounding box coordinates.
  * `labels_file` - path to file with class labels in csv format.
  * `images_dir` - path to images folder (Optional).
  * `label_start` - specifies label index start in label map. You can provide another value, if you want to use this dataset for separate label validation (Optional, default value is 1).

## <a name="customizing-dataset-meta"></a>Customizing Dataset Meta
There are situations when we need to customize some default dataset parameters (e.g. replace original dataset label map with own.)
You are able to overload parameters such as `label_map`, `segmentation_colors`, `background_label` using `dataset_meta_file` argument.
Dataset meta file is JSON file, which can contain the following parameters:
  * `label_map` is dictionary where `<CLASS_ID>` is key and `<CLASS_NAME>` - value.
  * `labels` is the list of strings, which represent class names (order is matter, the index of class name used as class id). Can be used instead `label_map`.
  * `background_label` - id of background label in the dataset.
  * `segmentation_colors` (if your dataset for semantic segmentation task uses color encoding). Segmentation colors is a list of channel-wise values for each class. (e.g. if your dataset has 3 classes in BGR colors, segmentation colors for it will looks like: `[[255, 0, 0], [0, 255, 0], [0, 0, 255]]`).
Example of dataset_meta.json content:
```json
{
"label_map": {"0": "background", "1": "cat", "2": "dog"},
"background_label": "0",
"segmentation_colors": [[0, 0, 0], [255, 0, 0], [0, 0, 255]]
}
```
