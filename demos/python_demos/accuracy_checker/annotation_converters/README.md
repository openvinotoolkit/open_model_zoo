# Annotation Converters

Annotation converter is a function which converts annotation file to suitable for metric evaluation format.
Each annotation converter expects specific annotation file format or data structure, which depends on original dataset.
If converter for your data format is not supported by Accuracy Checker, you can provide your own annotation converter.
Each annotation converter has parameters available for configuration.

The command line for annotation conversion looks like:

```bash
python3 convert_annotation.py <converter_name> <converter_specific parameters>
```
You may refer to `-h, --help` to full list of command line options. Some optional arguments are:

* `-o, --output_dir` - directory to save converted annotation and meta info.
* `-a, --annotation_name` - annotation file name.
* `-m, --meta_name` - meta info file name.

Accuracy Checker supports following list of annotation conveters and specific for them parameters:
* `wider` - converts from Wider Face dataset to `DetectionAnnotation`.
  * `annotation_file` - path to txt file, which contains ground truth data in WiderFace dataset format.
  * `label_start` - specifies face label index in label map. Default value is 1. You can provide another value, if you want to use this dataset for separate label validation,
  in case when your network predicts other class for faces.
* `sample` - converts annotation for SampleNet to `ClassificationAnnotation`.
  * `labels_file` - path to txt file which contains labels name.
* `voc07` - converts Pascal VOC 2007 annotation for detection task to `DetectionAnnotation`.
  * `devkit_dir` - path to VOC Devkit root directory.
* `voc_segmentation` - converts Pascal VOC annotation for semantic segmentation task to `SegmentationAnnotation`.
  * `devkit_dir` - path to VOC Devkit root directory.
* `detection_opencv_storage` - converts detection annotation stored in Detection OpenCV storage format to `DetectionAnnotation`.
  * `file_path` - path to annotation in xml format.
  * `image_names` - path to txt file, which contains image name list for dataset.
  * `label_start` - specifies label index start in label map. Default value is 1. You can provide another value, if you want to use this dataset for separate label validation.
  * `background_label` - specifies which index will be used for background label. You can not provide this parameter if your dataset has not background label
* `face_reid_pairwise` - converts Labeled Faces in the Wild dataset for face reidentification to `ReidentificationClassificationAnnotation`.
  * `pairs_file` - path to file with annotation positive and negative pairs.
  * `train_file` - path to file with annotation positive and negative pairs used for network train (optional parameter).
  * `landmarks_file` - path to file with facial landmarks coordinates for annotation images (optional parameter).
* `landmarks_regression` - converts VGG Face 2 dataset for facial landmarks regression task to `PointRegressionAnnotation`.
  * `landmarks_csv` - path to csv file with coordinates of landmarks points.
  * `bbox_csv` - path to cvs file which contains bounding box coordinates for faces (optional parameter).
