# Postprocessors

Postprocessor is function which processes prediction and/or annotation data after model infer and before metric calculation. For correct work postprocessors require specific representation format.
(e. g. clip boxes postprocessor expects detection annotation and detection prediction for processing).

In case when you use complicated representation located in representation container, you can add options `annotation_source` and `prediction_source` in configuration file,
if you want process only specific representations, another way postprocessor will be used for all suitable representations. `annotation_source` and `prediction_source` should contain
comma separated list of annotation identifiers and output layer names respectively.

Every postprocessor has parameters available for configuration.

Accuracy Checker supports following set of postprocessors:

* `cast_to_int` - casting detection bounding box coordinates given in floating point format to integer. Supported representations: `DetectionAnotation`, `DetectionPrediction`, `TextDetectionAnnotation`, `TextDetectionPrediction`.
  * `round_policy` - method for rounding: `nearest`, `greater`, `lower`, `nearest_to_zero`.
* `clip_boxes` - clipping detection bounding box sizes. Supported representations: `DetectionAnotation`, `DetectionPrediction`.
  * `dst_width` and `dst_height` - destination width and height for box clipping respectively. You can also use `size` instead in case when destination sizes are equal. If not provided, image size will be used.
  * `apply_to` - option which determines target boxes for processing (`annotation` for ground truth boxes and `prediction` for detection results, `all` for both).
  * `bboxes_normalized` is flag which says that target bounding boxes are in normalized format.
* `normalize_boxes` - normalizing ground truth detection bounding boxes (cast to range [0, 1]). Supported representations: `DetectionAnotation`, `DetectionPrediction`, `ActionDetectionAnotation`, `ActionDetectionPrediction`
* `correct_yolo_v2_boxes` - resizing detection prediction bbox coordinates using specific for Yolo v2 approach. Supported representations: `DetectionAnotation`, `DetectionPrediction`.
  * `dst_width` and `dst_height` - destination width and height respectively. You can also use `size` instead in case when destination sizes are equal.
* `resize_prediction_boxes` - resizing normalized detection prediction boxes according to image size. Supported representations: `DetectionAnotation`, `DetectionPrediction`.
* `faster_rcnn_postprocessing_resize` - resizing normalized detection prediction boxes according to the original image size before preprocessing steps.
    Supported representations: `DetectionAnotation`, `DetectionPrediction`.
    At the moment works in the following cases only:
   - the preprocessing steps contains only one operation changing input image size, and the operation is `resize`
   - the preprocessing steps contains only two operations changing input image size, and the operations are `resize` and then `padding`.
* `nms` - non-maximum suppression. Supported representations: `DetectionAnotation`, `DetectionPrediction`, `ActionDetectionAnnotation`, `ActionDetectionPrediction`.
  * `overlap` - overlap threshold for merging detections.
* `soft_nms` - soft non-maximum suppression. Supported representations: `DetectionAnotation`, `DetectionPrediction`, `ActionDetectionAnnotation`, `ActionDetectionPrediction`.
  * `keep_top_k`  - the maximal number of detections which should be kept.
  * `sigma` - sigma-value for updated detection score calculation.
  * `min_score` - break point.
* `filter` - filtering data using different parameters. Supported representations: `DetectionAnotation`, `DetectionPrediction`.
  * `apply_to` - determines target boxes for processing (`annotation` for ground truth boxes and `prediction` for detection results, `all` for both).
  * `remove_filtered` - removing filtered data. Annotations support ignoring filtered data without removing as default, in other cases filtered data will be removed automatically.
  * Supported parameters for filtering: `labels`, `min_confidence`, `height_range`, `width_range`, `is_empty`, `min_visibility`, `aspect_ratio`, `area_ratio`, `area_range`.
  Filtering by `height_range`, `width_range` are also available for `TextDetectionAnnotation`, `TextDetectionPrediction`, `area_range`  - for `PoseEstimationAnnotation`, `PoseEstimationPrediction` and `TextDetectionAnnotation`, `TextDetectionPrediction`.
* `normalize_landmarks_points` - normalizing ground truth landmarks points. Supported representations: `FacialLandmarksAnnotation`, `FacialLandmarksPrediction`.
  * `use_annotation_rect` - allows to use size of rectangle saved in annotation metadata for point scaling instead source image size.
 `encode_segmentation_mask` - encoding segmentation label image as segmentation mask. Supported representations: `SegmentationAnotation`, `SegmentationPrediction`.
  * `apply_to` - determines target masks for processing (`annotation` for ground truth and `prediction` for detection results, `all` for both).
  **Note:** this postprocessing requires specific dataset meta: `segmentation_colors` for annotations and `prediction_to_gt_labels` for predictions.
* `resize_segmentation_mask` - resizing segmentation mask. Supported representations: `SegmentationAnotation`, `SegmentationPrediction`.
  * `dst_width` and `dst_height` - destination width and height for resize respectively. You can also use `size` instead in case when destination sizes are equal.
    If any of these parameters are not specified, image size will be used as default.
  * `apply_to` - determines target masks for processing (`annotation` for ground truth and `prediction` for detection results, `all` for both).
* `extend_segmentation_mask` - extending annotation segmentation mask to predicted mask size making border filled by specific value. Supported representations: `SegmentationAnotation`, `SegmentationPrediction`.
  * `filling_label` - value for filling border (default 255).
* `zoom_segmentation_mask` - zooming segmentation mask. Supported representations: `SegmentationAnotation`, `SegmentationPrediction`.
  * `zoom` - size for zoom operation.
* `crop_segmentation_mask` - cropping 3-d annotation mask. Supported representations: `BrainTumorSegmentationAnnotation`, `BrainTumorSegmentationPrediction`.
  * `dst_width`, `dst_height` and `dst_volume` are destination width, height and volume for cropped 3D-image respectively.
    You can also use `size` instead in case when destination sizes are equal for all three dimensions.
* `clip_segmentation_mask` - clipping segmentation mask values. Supported representations: `BrainTumorSegmentationAnnotation`, `BrainTumorSegmentationPrediction`.
  * `min_value` - lower bound of range.
  * `max_value` - upper bound of range.
* `segmentation_prediction_resample` - resamples output prediction in two steps: 1) resizes it to bounding box size; 2) extends to annotation size. Supported representations: `BrainTumorSegmentationAnnotation`, `BrainTumorSegmentationPrediction`. For correct bounding box size must be set via tag `boxes_file` in `brats_numpy` [converter](../annotation_converters/README.md) or `crop_brats` [preprocessor](../preprocessor/README.md).
  * `make_argmax` - applies argmax operation to prediction mask after resampling (by default `False`). Must be specified only one option `make_argmax`.
* `transform_brats_prediction` - transforms prediction from `WT-TC-ET` format to `NCR/NET-ED-ET`. Sequentially fills one-channel mask with specified `values` for elements passing the threshold (threshold is `0.5`) from each prediction channel in specified `order`.
  * `order` - specifies filling order for channels
  * `values` - specifies values for each channel according to new order
* `extract_prediction_answers` - extract predicted sequence of tokens from annotation text. Supported representations: `QuestionAnsweringAnnotation`, `QuestionAnsweringPrediction`.
  * `max_answer` - maximum answer length (Optional, default value is 30).
  * `n_best_size` - total number of n-best prediction size for the answer (Optional, default value is 20).
* `translate_3d_poses` - translating 3D poses. Supported representations: `PoseEstimation3dAnnotation`, `PoseEstimation3dPrediction`. Shifts 3D coordinates of each predicted poses on corresponding translation vector.
* `resize_super_resolution` - resizing super resolution predicted image. Supported representations: `SuperResolutionAnotation`, `SuperResolutionPrediction`.
  * `dst_width` and `dst_height` - destination width and height for resizing respectively. You can also use `size` instead in case when destination sizes are equal.
    If any of these parameters are not specified, gt high resolution image size will be used as default.
* `resize_style_transfer` - resizing style transfer predicted image. Supported representations: `StyleTransferAnotation`, `StyleTransferPrediction`.
  * `dst_width` and `dst_height` - destination width and height for resizing respectively.
* `crop_ground_truth_image` - croping ground truth image. Supported representations: `ImageInpaintingAnnotation`.
