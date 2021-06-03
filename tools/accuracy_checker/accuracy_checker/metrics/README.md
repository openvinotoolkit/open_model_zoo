# Metrics

Metric is a class which is used to compare predicted data with annotated data and perform quality measurement.
For correct work metrics require specific representation format.
(e. g. map expects detection annotation and detection prediction for evaluation).

In case when you use complicated representation located in representation container, you need to add options `annotation_source` and `prediction_source` in configuration file to
select specific representation, another way metric calculation possible only if container has only one suitable representation and will be resolved automatically.
`annotation_source` and `prediction_source` should contain only one annotation identifier and output layer name respectively.
You may optionally provide `reference` field for metric, if you want calculated metric tested against specific value (i.e. reported in canonical paper) and acceptable `abs_threshold` and `rel_threshold` for absolute and relative metric deviation from reference value respectively.

Every metric has parameters available for configuration. The metric and its parameters are set through the configuration file. Metrics are provided in `datasets` section of configuration file to use specific metric.

## Supported Metrics

Accuracy Checker supports following set of metrics:

* `accuracy` - classification accuracy metric, defined as the number of correct predictions divided by the total number of predictions. Metric is calculated as a percentage. Direction of metric's growth is higher-better.
Supported representations: `ClassificationAnnotation`, `TextClassificationAnnotation`, `ClassificationPrediction`, `ArgMaxClassificationPrediction`.
  * `top_k` - the number of classes with the highest probability, which will be used to decide if prediction is correct.
  * `match` - Batch-oriented binary classification metric. Metric value calculated for each record in batch. Parameter `top_k` ignored in this mode.
* `accuracy_per_class` - classification accuracy metric which represents results for each class.  Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `ClassificationAnnotation`, `ClassificationPrediction`.
  * `top_k` - the number of classes with the highest probability, which will be used to decide if prediction is correct.
  * `label_map` - the field in annotation metadata, which contains dataset label map (Optional, should be provided if different from default).
* `character_recognition_accuracy` - accuracy metric for character recognition task. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `CharacterRecognitionAnnotation`, `CharacterRecognitionPrediction`.
  * `remove_spaces` - allow removement spaces from reference and predicted strings (Optional, default - `False`).
* `label_level_recognition_accuracy` - [label level recognition accuracy](https://dl.acm.org/doi/abs/10.1145/1143844.1143891) metric for text line character recognition task using [editdistance](https://pypi.org/project/editdistance/). Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `CharacterRecognitionAnnotation`, `CharacterRecognitionPrediction`.
* `classification_f1-score` - [F1 score](https://en.wikipedia.org/wiki/F1_score) metric for classification task. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `ClassificationAnnotation`, `TextClassificationAnnotation`, `ClassificationPrediction`.
  * `label_map` - the field in annotation metadata, which contains dataset label map (Optional, should be provided if different from default).
* `metthews_correlation_coef` - [Matthews correlation coefficient (MCC)](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient) for binary classification. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `ClassificationAnnotation`, `TextClassificationAnnotation`, `ClassificationPrediction`.
* `roc_auc_score` - [ROC AUC score](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) for binary classification. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `ClassificationAnnotation`, `TextClassificationAnnotation`, `ClassificationPrediction` `ArgMaxClassificationPrediction`.
* `acer_score` - metric for the classification tasks. Can be obtained from the following formula: `ACER = (APCER + BPCER)/2 = ((fp / (tn + fp)) + (fn / (fn + tp)))/2`. For more details about metrics see the section 9.3: <https://arxiv.org/abs/2007.12342>. Metric is calculated as a percentage. Direction of metric's growth is higher-worse. Supported representations: `ClassificationAnnotation`, `TextClassificationAnnotation`, `ClassificationPrediction`.
* `clip_accuracy` - classification video-level accuracy metric. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `ClassificationAnnotation`, `ClassificationPrediction`.
* `map` - mean average precision. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `DetectionAnnotation`, `DetectionPrediction`.
  * `overlap_threshold` - minimal value for intersection over union that allows to make decision that prediction bounding box is true positive.
  * `overlap_method` - method for calculation bbox overlap. You can choose between intersection over union (`iou`), defined as area of intersection divided by union of annotation and prediction boxes areas, and intersection over area (`ioa`), defined as area of intersection divided by ara of prediction box.
  * `include_boundaries` - allows include boundaries in overlap calculation process. If it is True then width and height of box is calculated by max - min + 1.
  * `ignore_difficult` - allows to ignore difficult annotation boxes in metric calculation. In this case, difficult boxes are filtered annotations from postprocessing stage.
  * `distinct_conf` - select only values for distinct confidences.
  * `allow_multiple_matches_per_ignored` - allows multiple matches per ignored.
  * `label_map` - the field in annotation metadata, which contains dataset label map  (Optional, should be provided if different from default).
  * `integral` - integral type for average precision calculation. Pascal VOC `11point` and `max` approaches are available.
* `miss_rate` - miss rate metric of detection models. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `DetectionAnnotation`, `DetectionPrediction`.
  * `overlap_threshold` - minimal value for intersection over union that allows to make decision that prediction bounding box is true positive.
  * `overlap_method` - method for calculation bbox overlap. You can choose between intersection over union (`iou`), defined as area of intersection divided by union of annotation and prediction boxes areas, and intersection over area (`ioa`), defined as area of intersection divided by ara of prediction box.
  * `include_boundaries` - allows include boundaries in overlap calculation process. If it is True then width and height of box is calculated by max - min + 1.
  * `ignore_difficult` - allows to ignore difficult annotation boxes in metric calculation. In this case, difficult boxes are filtered annotations from postprocessing stage.
  * `distinct_conf` - select only values for distinct confidences.
  * `allow_multiple_matches_per_ignored` - allows multiple matches per ignored.
  * `label_map` - the field in annotation metadata, which contains dataset label map  (Optional, should be provided if different from default).
  * `fppi_level` - false positive per image level.
* `recall` - recall metric of detection models. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `DetectionAnnotation`, `DetectionPrediction`.
  * `overlap_threshold` - minimal value for intersection over union that allows to make decision that prediction bounding box is true positive.
  * `overlap_method` - method for calculation bbox overlap. You can choose between intersection over union (`iou`), defined as area of intersection divided by union of annotation and prediction boxes areas, and intersection over area (`ioa`), defined as area of intersection divided by ara of prediction box.
  * `include_boundaries` - allows include boundaries in overlap calculation process. If it is True then width and height of box is calculated by max - min + 1.
  * `ignore_difficult` - allows to ignore difficult annotation boxes in metric calculation. In this case, difficult boxes are filtered annotations from postprocessing stage.
  * `distinct_conf` - select only values for distinct confidences.
  * `allow_multiple_matches_per_ignored` - allows multiple matches per ignored.
  * `label_map` - the field in annotation metadata, which contains dataset label map (Optional, should be provided if different from default).
* `detection_accuracy` - accuracy for detection models. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `DetectionAnnotation`, `DetectionPrediction`.
  * `overlap_threshold` - minimal value for intersection over union that allows to make decision that prediction bounding box is true positive.
  * `overlap_method` - method for calculation bbox overlap. You can choose between intersection over union (`iou`), defined as area of intersection divided by union of annotation and prediction boxes areas, and intersection over area (`ioa`), defined as area of intersection divided by ara of prediction box.
  * `include_boundaries` - allows include boundaries in overlap calculation process. If it is True then width and height of box is calculated by max - min + 1.
  * `label_map` - the field in annotation metadata, which contains dataset label map  (Optional, should be provided if different from default).
  * `use_normalization` - allows to normalize confusion_matrix for metric calculation.
* `segmentation_accuracy` - pixel accuracy for semantic segmentation models. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `SegmentationAnnotation`, `SegmentationPrediction`.
  * `use_argmax` - allows to use argmax for prediction mask.
  * `ignore_label` - specified which class_id prediction should be ignored during metric calculation. (Optional, if not provided, all labels will be used).
* `mean_iou` - mean intersection over union for semantic segmentation models. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `SegmentationAnnotation`, `SegmentationPrediction`.
  * `use_argmax` - allows to use argmax for prediction mask.
  * `ignore_label` - specified which class_id prediction should be ignored during metric calculation. (Optional, if not provided, all labels will be used).
* `mean_accuracy` - mean accuracy for semantic segmentation models. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `SegmentationAnnotation`, `SegmentationPrediction`.
  * `use_argmax` - allows to use argmax for prediction mask.
  * `ignore_label` - specified which class_id prediction should be ignored during metric calculation. (Optional, if not provided, all labels will be used).
* `frequency_weighted_accuracy` - frequency weighted accuracy for semantic segmentation models. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `SegmentationAnnotation`, `SegmentationPrediction`.
  * `use_argmax` - allows to use argmax for prediction mask.
  * `ignore_label` - specified which class_id prediction should be ignored during metric calculation. (Optional, if not provided, all labels will be used).
More detailed information about calculation segmentation metrics you can find [here](https://arxiv.org/abs/1411.4038).
* `cmc` - Cumulative Matching Characteristics (CMC) score. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `ReIdentificationAnnotation`, `ReIdentificationPrediction`.
  * `top_k` -  number of k highest ranked samples to consider when matching.
  * `separate_camera_set` - should identities from the same camera view be filtered out.
  * `single_gallery_shot` -  each identity has only one instance in the gallery.
  * `number_single_shot_repeats` - number of repeats for single_gallery_shot setting (required for CUHK).
  * `first_match_break` - break on first matched gallery sample.
* `reid_map` - Mean Average Precision score for object reidentification. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `ReIdentificationAnnotation`, `ReIdentificationPrediction`.
  * `uninterpolated_auc` - should area under precision recall curve be computed using trapezoidal rule or directly.
* `pairwise_accuracy` - pairwise accuracy for object reidentification. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `ReIdentificationClassificationAnnotation`, `ReIdentificationPrediction`.
  * `min_score` - min score for determining that objects are different. You can provide value or use `train_median` or `best_train_threshold` values which will be calculated if annotations has training subset.
  * `distance_method` - allows to choose one of the distance calculation methods (optional, supported methods are `euclidian_distance` and `cosine_distance`, default - `euclidian_distance`).
  * `subtract_mean` - allows to subtract mean calculated on train embeddings before calculating the distance(optional, default - `False`).
* `pairwise_accuracy_subsets` - object reidentification pairwise accuracy with division dataset on test and train subsets for calculation mean score. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `ReIdentificationClassificationAnnotation`, `ReIdentificationPrediction`.
  * `subset_number` - number of subsets for separating.
  * `min_score` - min score for determining that objects are different. You can provide value or use `train_median` or `best_train_threshold` values which will be calculated if annotations has training subset.
  * `distance_method` - allows to choose one of the distance calculation methods (optional, supported methods are `euclidian_distance` and `cosine_distance`, default - `euclidian_distance`).
  * `subtract_mean` - allows to subtract mean calculated on train embeddings before calculating the distance(optional, default - `False`).
* `localization_recall` - recall metric used for evaluation place recognition task. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `PlaceRecognitionAnnotation`, `ReidentificationPrediction`.
  * `top_k` - number of k highest ranked samples to consider when matching.
  * `distance_threshold` - distance threshold for search positive matching pairs between query and gallery (Optional, default 25).
* `mae` - [Mean Absolute Error](https://en.wikipedia.org/wiki/Mean_absolute_error). Direction of metric's growth is higher-worse. Supported representations: `RegressionAnnotation`, `RegressionPrediction`, `FeatureRegressionAnnotation`, `DepthEstimationAnnotation`, `DepthEstimationPrediction`, `ImageProcessingAnnotation`, `ImageProcessingPrediction`, `BackgroundMattingAnnotation`, `BackgroundMattingPrediction`.
  * `max_error` - allow to calculate maximal error in range. Optional, default `False`.
* `mae_on_intervals` - Mean Absolute Error estimated magnitude for specific value range. Direction of metric's growth is higher-worse. Supported representations: `RegressionAnnotation`, `RegressionPrediction`.
  * `intervals` - comma-separated list of interval boundaries.
  * `ignore_values_not_in_interval` - allows create additional intervals for values less than minimal value in interval and greater than maximal.
  * `start` , `step`, `end` - way to generate range of intervals from `start` to `end` with length `step`.
* `mse` - [Mean Squared Error](https://en.wikipedia.org/wiki/Mean_squared_error). Direction of metric's growth is higher-worse. Supported representations: `RegressionAnnotation`, `RegressionPrediction`, `FeatureRegressionAnnotation`, `DepthEstimationAnnotation`, `DepthEstimationPrediction`, `ImageProcessingAnnotation`, `ImageProcessingPrediction`, `BackgroundMattingAnnotation`, `BackgroundMattingPrediction`.
  * `max_error` - allow to calculate maximal error in range. Optional, default `False`.
* `mse_on_intervals` - Mean Squared Error estimated magnitude for specific value range. Direction of metric's growth is higher-worse. Supported representations: `RegressionAnnotation`, `RegressionPrediction`.
  * `intervals` - comma-separated list of interval boundaries.
  * `ignore_values_not_in_interval` - allows create additional intervals for values less than minimal value in interval and greater than maximal.
  * `start`, `step`, `end` - generate range of intervals from `start` to `end` with length `step`.
* `rmse` - [Root Mean Squared Error](https://en.wikipedia.org/wiki/Root-mean-square_deviation). Direction of metric's growth is higher-worse. Supported representations: `RegressionAnnotation`, `RegressionPrediction`, `FeatureRegressionAnnotation`, `DepthEstimationAnnotation`, `DepthEstimationPrediction`, `ImageProcessingAnnotation`, `ImageProcessingPrediction`, `BackgroundMattingAnnotation`, `BackgroundMattingPrediction`.
  * `max_error` - allow to calculate maximal error in range. Optional, default `False`.
* `rmse_on_intervals` - Root Mean Squared Error estimated magnitude for specific value range. Direction of metric's growth is higher-worse. Supported representations: `RegressionAnnotation`, `RegressionPrediction`.
  * `intervals` - comma-separated list of interval boundaries.
  * `ignore_values_not_in_interval` - allows create additional intervals for values less than minimal value in interval and greater than maximal.
  * `start`, `step`, `end` - generate range of intervals from `start` to `end` with length `step`.
* `mape` - [Mean Absolute Percentage Error](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error). Direction of metric's growth is higher-worse. Supported representations: `RegressionAnnotation`, `RegressionPrediction`, `FeatureRegressionAnnotation`, `DepthEstimationAnnotation`, `DepthEstimationPrediction`, `ImageProcessingAnnotation`, `ImageProcessingPrediction`, `BackgroundMattingAnnotation`, `BackgroundMattingPrediction`.
  * `max_error` - allow to calculate maximal error in range. Optional, default `False`.
* `log10_error` - Logarithmic mean absolute error. Direction of metric's growth is higher-worse. Supported representations: `RegressionAnnotation`, `RegressionPrediction`, `FeatureRegressionAnnotation`, `DepthEstimationAnnotation`, `DepthEstimationPrediction`.
  * `max_error` - allow to calculate maximal error in range. Optional, default `False`.
* `per_point_normed_error` - Normed Error for measurement the quality of landmarks' positions. Estimated results for each point independently. Direction of metric's growth is higher-worse. Supported representations: `FacialLandmarksAnnotation`, `FacialLandmarksPrediction`, `FacialLandmarks3DAnnotation`, `FacialLandmarks3DPrediction`.
* `normed_error` - Normed Error for measurement the quality of landmarks' positions. Direction of metric's growth is higher-worse. Supported representations: `FacialLandmarksAnnotation`, `FacialLandmarksPrediction`, `FacialLandmarks3DAnnotation`, `FacialLandmarks3DPrediction`.
  * `calculate_std` - allows calculation of standard deviation (default value: `False`).
  * `percentile` - calculate error rate for given percentile.
* `nme` - Mean Normed Error for measurement quality of landmarks positions. Direction of metric's growth is higher-worse. Supported representations: `FacialLandmarks3DAnnotation`, `FacialLandwarks3DPrediction`.
  * `only_2d` - evaluate metric only for 2d case.
* `psnr` - [Peak signal to noise ratio](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio). Metric is calculated as a decibel(dB). Direction of metric's mean growth is higher-better. Direction of metric's std and max_error growth is higher-worse. Supported representations: `SuperResolutionAnnotation`, `SuperResolutionPrediction`, `ImageProcessingAnnotation`, `ImageProcessingPrediction`, `ImageInpaintingAnnotation`, `ImageInpaintingPrediction`.
  * `color_order` - the field specified which color order `BGR` or `RGB` will be used during metric calculation (Optional. Default value is RGB), used only if you have 3-channel images.
  * `normalized_images` - whether the images are normalized in [0, 1] range or not. Optional, default `False`.
* `psnr-b` - [Peak signal to noise ratio with blocked effect factor](https://doi.org/10.1007/978-3-642-34595-1_16). Metric is calculated as a decibel(dB). Direction of metric's mean growth is higher-better. Direction of metric's std and max_error growth is higher-worse. Supported representations: `SuperResolutionAnnotation`, `SuperResolutionPrediction`, `ImageProcessingAnnotation`, `ImageProcessingPrediction`, `ImageInpaintingAnnotation`, `ImageInpaintingPrediction`.
  * `color_order` - the field specified which color order `BGR` or `RGB` will be used during metric calculation (Optional. Default value is RGB), used only if you have 3-channel images.
  * `normalized_images` - whether the images are normalized in [0, 1] range or not. Optional, default `False`.
  * `block_size` - block size for blocked effect factor. Optional, default `8`.
  * `max_error` - allow to calculate maximal error in range. Optional, default `False`.
* `vif` - [Visual Information Fidelity](https://en.wikipedia.org/wiki/Visual_Information_Fidelity). Direction of metric's mean growth is higher-better. Direction of metric's std and max_error growth is higher-worse. Supported representations: `SuperResolutionAnnotation`, `ImageInpaintingAnnotation`, `ImageProcessingAnnotation`, `StyleTransferAnnotation`, `SuperResolutionPrediction`, `ImageInpaintingPrediction`, `ImageProcessingPrediction`, `StyleTransferPrediction`.
  * `sigma_nsq` - variance of the visual noise (default = 2).
  * `max_error` - allow to calculate maximal error in range. Optional, default `False`.
* `lpips` - [Learned Perceptual Image Patch Similarity](https://richzhang.github.io/PerceptualSimilarity/). Direction of metric's growth is higher-worse. Supported representations: `SuperResolutionAnnotation`, `SuperResolutionPrediction`, `ImageProcessingAnnotation`, `ImageProcessingPrediction`, `ImageInpaintingAnnotation`, `ImageInpaintingPrediction`.
** Metric calculation requires `lpips` package installation.**
  * `color_order` - the field specified which color order `BGR` or `RGB` will be used during metric calculation (Optional. Default value is RGB), used only if you have 3-channel images.
  * `normalized_images` - whether the images are normalized in [0, 1] range or not. Optional, default `False`.
  * `net` - network for perceptual loss calculation. Supported models: `alex` - for AlexNet, `vgg` - for VGG16, `squeeze` - for SqueezeNet1.1. Optional, default `alex`.
  * `distance_threshold` - distance threshold for getting images ratio with greater distance. Optional, if not provided, this coefficient will not be calculated.
  * `max_error` - allow to calculate maximal error in range. Optional, default `False`.
* `ssim` - [Structural similarity](https://en.wikipedia.org/wiki/Structural_similarity). Direction of metric's mean growth is higher-better. Direction of metric's std and max_error growth is higher-worse. Supported representations: `ImageProcessingAnnotation`, `ImageProcessingPrediction`, `ImageInpaintingAnnotation`, `ImageInpaintingPrediction`, `SuperResolutionAnnotation`, `SuperResolutionPrediction`.
  * `max_error` - allow to calculate maximal error in range. Optional, default `False`.
* `angle_error` - Mean angle error and Standard deviation of angle error for gaze estimation. Direction of metric's growth is higher-worse. Supported representations: `GazeVectorAnnotation`, `GazeVectorPrediction`.
  * `max_error` - allow to calculate maximal error in range. Optional, default `False`.
* `relative_l2_error` - Mean relative error defined like L2 norm for difference between annotation and prediction normalized by L2 norm for annotation value. Direction of metric's growth is higher-worse. Supported representations:
  `FeatureRegressionAnnotation`, `RegressionPrediction`.
  * `max_error` - allow to calculate maximal error in range. Optional, default `False`.
* `multi_accuracy` - accuracy for multilabel recognition task. Direction of metric's growth is higher-better. Supported representations: `MultiLabelRecognitionAnnotation`, `MultiLabelRecognitionPrediction`.
  * `label_map` - the field in annotation metadata, which contains dataset label map (Optional, should be provided if different from default).
  * `calculate_average` - allows calculation of average accuracy (default value: `True`).
* `multi_precision` - precision metric for multilabel recognition. Direction of metric's growth is higher-better. Supported representations: `MultiLabelRecognitionAnnotation`, `MultiLabelRecognitionPrediction`.
  * `label_map` - the field in annotation metadata, which contains dataset label map  (Optional, should be provided if different from default).
  * `calculate_average` - allows calculation of average precision (default value: `True`).
* `multi_recall` - recall metric for multilabel recognition. Direction of metric's growth is higher-better. Supported representations: `MultiLabelRecognitionAnnotation`, `MultiLabelRecognitionPrediction`.
  * `label_map` - the field in annotation metadata, which contains dataset label map  (Optional, should be provided if different from default).
  * `calculate_average` - allows calculation of average recall (default value: `True`).
* `f1_score` - [F score](https://en.wikipedia.org/wiki/F1_score) metric for multilabel recognition. Direction of metric's growth is higher-better. Supported representations: `MultiLabelRecognitionAnnotation`, `MultiLabelRecognitionPrediction`.
  * `label_map` - the field in annotation metadata, which contains dataset label map  (Optional, should be provided if different from default).
  * `calculate_average` - allows calculation of average f-score (default value: `True`).
* `focused_text_hmean` - Harmonic mean of precision and recall for focused scene text detection task introduced in [Robust Reading Competition challenge 2](https://rrc.cvc.uab.es/?ch=2&com=introduction). Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `TextDetectionAnnotation`, `TextDetectionPrediction`, `DetectionPrediction`.
  * `ignore_difficult` - allows to ignore difficult ground truth text polygons in metric calculation.
  * `area_precision_constrain` - minimal value for precision that allows to make decision that prediction polygon matched with annotation.
  * `area_recall_constrain` - minimal value for recall that allows to make decision that prediction polygon matched with annotation.
  * `center_diff_threshold` - acceptable difference between center of predicted text region and ground truth.
  * `one_to_one_match_score` - weight for one to one matching results (Optional, default 1).
  * `one_to_many_match_score` - weight for one to many matching results (Optional, default 0.8).
  * `many_to_one_match_score` - weight for many to one matching results (optional, default 1).
* `focused_text_precision` - precision for focused scene text detection task introduced in [Robust Reading Competition challenge 2](https://rrc.cvc.uab.es/?ch=2&com=introduction). Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `TextDetectionAnnotation`, `TextDetectionPrediction`, `DetectionPrediction`.
  * `ignore_difficult` - allows to ignore difficult ground truth text polygons in metric calculation.
  * `area_precision_constrain` - minimal value for precision that allows to make decision that prediction polygon matched with annotation.
  * `area_recall_constrain` - minimal value for recall that allows to make decision that prediction polygon matched with annotation.
  * `center_diff_threshold` - acceptable difference between center of predicted text region and ground truth.
  * `one_to_one_match_score` - weight for one to one matching results (Optional, default 1).
  * `one_to_many_match_score` - weight for one to many matching results (Optional, default 0.8).
  * `many_to_one_match_score` - weight for many to one matching results (optional, default 1).
* `focused_text_recall` - recall for focused scene text detection task introduced in [Robust Reading Competition challenge 2](https://rrc.cvc.uab.es/?ch=2&com=introduction). Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `TextDetectionAnnotation`, `TextDetectionPrediction`, `DetectionPrediction`.
  * `ignore_difficult` - allows to ignore difficult ground truth text polygons in metric calculation.
  * `area_precision_constrain` - minimal value for precision that allows to make decision that prediction polygon matched with annotation.
  * `area_recall_constrain` - minimal value for recall that allows to make decision that prediction polygon matched with annotation.
  * `center_diff_threshold` - acceptable difference between center of predicted text region and ground truth.
  * `one_to_one_match_score` - weight for one to one matching results (Optional, default 1).
  * `one_to_many_match_score` - weight for one to many matching results (Optional, default 0.8).
  * `many_to_one_match_score` - weight for many to one matching results (optional, default 1).
* `incidental_text_hmean` - Harmonic mean of precision and recall for incidental scene text detection task introduced in [Robust Reading Competition challenge 4](https://rrc.cvc.uab.es/?ch=4&com=introduction). Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `TextDetectionAnnotation`, `TextDetectionPrediction`.
  * `ignore_difficult` - allows to ignore difficult ground truth text polygons in metric calculation.
  * `iou_constrain` - minimal value for intersection over union that allows to make decision that prediction polygon is true positive.
  * `area_precision_constrain` - minimal value for precision that allows to make decision that prediction polygon matched with ignored annotation.
* `incidental_text_precision` - precision for incidental scene text detection task introduced in [Robust Reading Competition challenge 4](https://rrc.cvc.uab.es/?ch=4&com=introduction). Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `TextDetectionAnnotation`, `TextDetectionPrediction`.
  * `ignore_difficult` - allows to ignore difficult ground truth text polygons in metric calculation.
  * `iou_constrain` - minimal value for intersection over union that allows to make decision that prediction polygon is true positive.
  * `area_precision_constrain` - minimal value for precision that allows to make decision that prediction polygon matched with ignored annotation.
* `incidental_text_recall` - recall for incidental scene text detection task introduced in [Robust Reading Competition challenge 4](https://rrc.cvc.uab.es/?ch=4&com=introduction). Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `TextDetectionAnnotation`, `TextDetectionPrediction`.
  * `ignore_difficult` - allows to ignore difficult ground truth text polygons in metric calculation.
  * `iou_constrain` - minimal value for intersection over union that allows to make decision that prediction polygon is true positive.
  * `area_precision_constrain` - minimal value for precision that allows to make decision that prediction polygon matched with ignored annotation.
* `coco_precision` - MS COCO Average Precision metric for keypoints recognition and object detection tasks. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `PoseEstimationAnnotation`, `PoseEstimationPrediction`, `DetectionAnnotation`, `DetectionPrediction`.
  * `max_detections` - max number of predicted results per image. If you have more predictions,the results with minimal confidence will be ignored.
  * `threshold` - intersection over union threshold. You can specify one value or comma separated range of values. This parameter supports precomputed values for standard COCO thresholds (`.5`, `.75`, `.5:.05:.95`).
* `coco_recall` - MS COCO Average Recall metric for keypoints recognition and object detection tasks. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `PoseEstimationAnnotation`, `PoseEstimationPrediction`, `DetectionAnnotation`, `DetectionPrediction`.
  * `max_detections` - max number of predicted results per image. If you have more predictions,the results with minimal confidence will be ignored.
  * `threshold` - intersection over union threshold. You can specify one value or comma separated range of values. This parameter supports precomputed values for standard COCO thresholds (`.5`, `.75`, `.5:.05:.95`).
* `coco_keypoints_precision` - MS COCO Average Precision metric for keypoints recognition task. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `PoseEstimationAnnotation`, `PoseEstimationPrediction`.
  * `max_detections` - max number of predicted results per image. If you have more predictions,the results with minimal confidence will be ignored.
  * `threshold` - intersection over union threshold. You can specify one value or comma separated range of values. This parameter supports precomputed values for standard COCO thresholds (`.5`, `.75`, `.5:.05:.95`).
* `coco_keypoints_recall` - MS COCO Average Recall metric for keypoints recognition task. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `PoseEstimationAnnotation`, `PoseEstimationPrediction`.
  * `max_detections` - max number of predicted results per image. If you have more predictions,the results with minimal confidence will be ignored.
  * `threshold` - intersection over union threshold. You can specify one value or comma separated range of values. This parameter supports precomputed values for standard COCO thresholds (`.5`, `.75`, `.5:.05:.95`).
* `coco_segm_precision` - MS COCO Average Precision metric for instance segmentation task. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `CoCoInstanceSegmentationAnnotation`, `CoCoInstanceSegmentationPrediction`.
  * `max_detections` - max number of predicted results per image. If you have more predictions,the results with minimal confidence will be ignored.
  * `threshold` - intersection over union threshold. You can specify one value or comma separated range of values. This parameter supports precomputed values for standard COCO thresholds (`.5`, `.75`, `.5:.05:.95`).
* `coco_segm_recall` - MS COCO Average Recall metric for instance segmentation task. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `CoCoInstanceSegmentationAnnotation`, `CoCoInstanceSegmentationPrediction`.
  * `max_detections` - max number of predicted results per image. If you have more predictions,the results with minimal confidence will be ignored.
  * `threshold` - intersection over union threshold. You can specify one value or comma separated range of values. This parameter supports precomputed values for standard COCO thresholds (`.5`, `.75`, `.5:.05:.95`).
* `hit_ratio` - metric for recommendation system evaluation. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `HitRatioAnnotation`, `HitRatioPrediction`.
  * `top_k` - definition of number elements in rank list (optional, default 10).
* `ndcg` - [Normalized Discounted Cumulative Gain](https://en.wikipedia.org/wiki/Discounted_cumulative_gain). Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `HitRatioAnnotation`, `HitRatioPrediction`.
  * `top_k` - definition of number elements in rank list (optional, default 10).
* `dice` - [Sørensen–Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient). Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `BrainTumorSegmentationAnnotation, BrainTumorSegmentationPrediction`.
* `dice_index` - [Sørensen–Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient). Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `BrainTumorSegmentationAnnotation, BrainTumorSegmentationPrediction`, `SegmentationAnnotation, SegmentationPrediction`. Supports result representation for multiple classes. Metric represents result for each class if `label_map` for used dataset is provided, otherwise it represents overall result. For `brats_numpy` converter file with labels set in `labels_file` tag.
  * `mean` - allows calculation mean value (default - `True`).
  * `median` - allows calculation median value (default - `False`).
* `dice_unet3d` - [Sørensen–Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient). Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `BrainTumorSegmentationAnnotation, BrainTumorSegmentationPrediction`.
Applied for models trained on brats data with labels in range (0, 1, 2, 3). The metric is quite similar to `dice_index` with only the difference that represents data for three statically defined labels:  1) `whole tumor` - aggregated data for labels (1, 2, 3) of the dataset; 2) `tumor core` - aggregated data for labels (2, 3) of the dataset; 3) `enhancing tumor` - data for label (3) of the dataset.
  * `mean` - allows calculation mean value (default - `True`).
  * `median` - allows calculation median value (default - `False`).
* `dice_oar3d` - [Sørensen–Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient). Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `OAR3DTilingSegmentationAnnotation, SegmentationPrediction`.
* `bleu` - [Bilingual Evaluation Understudy](https://en.wikipedia.org/wiki/BLEU). Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `MachineTranslationAnnotation`, `MachineTranslationPrediction`.
  * `smooth` - Whether or not to apply Lin et al. 2004 smoothing (Optional, default `False`).
  * `max_order` - Maximum n-gram order to use when computing BLEU score. (Optional, default 4).
  * `smooth_method` - The smoothing method to use. Supported values: `exp`, `floor`, `add-k`, `none` (Optional, default value is `exp` is `smooth` is enabled and `none` if not).
  * `smooth_value` - the value for smoothing for `floor` or `add-k` smoothing methods. (Optional, applicable only if specific smoothing methods selected, default values are 0 or 1 for `floor` and `add-k` methods respectively).
  * `lower_case` - convert annotation and prediction tokens to lower case (Optional, default `False`).
* `f1` - F1-score for question answering task. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `QuestionAnsweringAnnotation`, `QuestionAnsweringPrediction`.
* `exact_match` - Exact matching (EM) metric for question answering task. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `QuestionAnsweringAnnotation`, `QuestionAnsweringPrediction`.
* `qa_embedding_accuracy` - Right context detection accuracy metric for question answering task solved by question vs context embeddings comparison. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `QuestionAnsweringEmbeddingAnnotation`, `QuestionAnsweringEmbeddingPrediction`.
* `ner_accuracy` - Token-level accuracy used for Named Entity Recognition task. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `BERTNamedEntityRecognitionAnnotation`, `SequenceClassificationAnnotation`, `SequenceClassificationPrediction`.
   * `label_map` - the field in annotation metadata, which contains dataset label map  (Optional, should be provided if different from default).
* `ner_recall` - Token-level recall used for Named Entity Recognition task. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `BERTNamedEntityRecognitionAnnotation`, `SequenceClassificationAnnotation`, `SequenceClassificationPrediction`.
   * `label_map` - the field in annotation metadata, which contains dataset label map  (Optional, should be provided if different from default).
* `ner_precision` - Token-level precision for Named Entity Recognition task. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `BERTNamedEntityRecognitionAnnotation`, `SequenceClassificationAnnotation`, `SequenceClassificationPrediction`.
   * `label_map` - the field in annotation metadata, which contains dataset label map  (Optional, should be provided if different from default).
* `ner_f_score` - Token-level F-score used for Named Entity Recognition task. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `BERTNamedEntityRecognitionAnnotation`, `SequenceClassificationAnnotation`, `SequenceClassificationPrediction`.
   * `label_map` - the field in annotation metadata, which contains dataset label map  (Optional, should be provided if different from default).
* `mpjpe_multiperson` - [Mean Per Joint Position Error](http://vision.imar.ro/human3.6m/pami-h36m.pdf) extended for multi-person case. Metric is calculated as a millimeters(mm). Direction of metric's growth is higher-worse. Supported representations: `PoseEstimation3dAnnotation`, `PoseEstimation3dPrediction`. As the first step, correspondence between ground truth and prediction skeletons is established for each image. Then MPJPE is computed for each ground truth and prediction pair. The error is averaged over poses in each frame, then over all frames.
* `face_recognition_tafa_pair_metric` - accuracy for face recognition models based on dot product of embedding values. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `ReIdentificationAnnotation`, `ReIdentificationPrediction`.
  * `threshold` - minimal dot product value of embeddings to identify as matched face pair.
* `youtube_faces_accuracy` - accuracy for face detection models calculated based on IOU values of ground truth bounding boxes and model-detected bounding boxes. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `DetectionAnnotation`, `DetectionPrediction`.
  * `overlap` - minimum IOU threshold to consider as a true positive candidate face.
  * `relative_size` - size of detected face candidate\'s area in proportion to the size of ground truth\'s face size. This value is set to filter candidates that have high IOU but have a relatively smaller face size than ground truth face size.
* `normalized_embedding_accuracy` - accuracy for re-identification models based on normalized dot product of embedding values. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `ReIdentificationAnnotation`, `ReIdentificationPrediction`.
  * `top_k` - the number of people with the highest probability, which will be used to decide if prediction is correct.
* `attribute_accuracy` - accuracy for attributes in attribute classification models. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `ContainerAnnotation` with `ClassificationAnnotation` and `ContainerPrediction` with `ClassificationPrediction`.
  * `attributes`: names of attributes.
  * `calculate_average` - allows calculation of average accuracy (default value: `True`).
* `attribute_recall` - recall for attributes in attribute classification models. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `ContainerAnnotation` with `ClassificationAnnotation` and `ContainerPrediction` with `ClassificationPrediction`.
  * `attributes`: names of attributes.
  * `calculate_average` - allows calculation of average recall (default value: `True`).
* `attribute_precision` - precision for attributes in attribute classification models. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `ContainerAnnotation` with `ClassificationAnnotation` and `ContainerPrediction` with `ClassificationPrediction`.
  * `attributes`: names of attributes.
  * `calculate_average` - allows calculation of average precision (default value: `True`).
* `wer` - Word error rate ([WER](https://en.wikipedia.org/wiki/Word_error_rate)). Metric is calculated as a percentage. Direction of metric's growth is higher-worse. Supported representations: `CharacterRecognitionAnnotation`, `CharacterRecognitionPrediction`.
* `cer` - Character error rate, character-level counterpart of [WER](https://en.wikipedia.org/wiki/Word_error_rate). Metric is calculated as a percentage. Direction of metric's growth is higher-worse. Supported representations: `CharacterRecognitionAnnotation`, `CharacterRecognitionPrediction`.
* `ser` - Sentence error rate (SER), which indicates the percentage of sentences, whose translations have not matched in an exact manner those of reference. Direction of metric's growth is higher-worse. Supported representations: `CharacterRecognitionAnnotation`, `CharacterRecognitionPrediction`.
* `score_class_comparison` - allows calculate an accuracy of quality score class(low/normal/good). It sorts all quality scores from annotations and predictions and set the k1 highest scores as high class and the k2 lowest scores as low class where k1 is `num_high_quality` and k2 is `num_low_quality`. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `QualityAssessmentAnnotation`, `QualityAssessmentPrediction`.
  * `num_high_quality` - the number of high class in total (default value: `1`).
  * `num_low_quality` - the number of low class in total (default value: `1`).
* `im2latex_images_match` - This metric gets formulas in `CharacterRecognitionAnnotation` and `CharacterRecognitionPrediction` format and based on this given text renders images with this formulas. Then for every two corresponding formulas images are getting compared. The result accuracy is percent of the equivalent formulas. Direction of metric's growth is higher-better.
  >Note: this metric requires installed packages texlive and imagemagick. In linux you can do it this way:
  > `sudo apt-get update && apt-get install texlive imagemagick`
* `pckh` - Percentage of Correct Keypoints normalized by head size. A detected joint is considered correct if the distance between the predicted and the true joint is within a certain threshold. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `PoseEstimationAnnotation`, `PoseEstimationPrediction`.
  * `threshold` - distance threshold (Optional, default 0.5).
  * `scale_bias` - bias for scale head size (Optional, default 0.6).
* `dna_seq_accuracy` - accuracy for DNA sequencing task. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `DNASequenceAnnotation`, `DNASequencePrediction`, `CharacterRecognitionAnnotation`, `CharacterRecognitionPrediction`.
  * `min_coverage` - minimum sequence coverage between predicted sequence and reference for positive measurement (Optional, default 0.5).
  * `balansed` - balanced accuracy metric (Optional, default false).
* `inception_score` - [Inception score](https://arxiv.org/abs/1801.01973) for generated images by GAN models. Direction of metric's growth is higher-worse. Supported representations: `ImageProcessingAnnotation`, `ImageProcessingPrediction`.
  * `eps` - epsilon to avoid nan during calculate log for metric.
  * `length` - length of input feature vector for metric.
* `fid` - Frechet Inception Distance to measure the distance between the distributions of synthesized images and real images. Direction of metric's growth is higher-worse. Supported representations: `ImageProcessingAnnotation`, `ImageProcessingPrediction`.
  * `eps` - epsilon to avoid nan during calculate sqrtm for metric.
  * `length` - length of input feature vector for metric.
* `epe` - Average End Point Error (EPE) metric for optical flow estimation task, defined as Euclidean distance between ground truth and predicted flow. Direction of metric's growth is higher-worse. Supported representations: `OpticalFlowAnnotation`, `OpticalFlowPrediction`.
  * `max_error` - allow to calculate maximal error in range. Optional, default `False`.
* `salience_mae` - Mean Absolute Error for salient object detection task. Direction of metric's growth is higher-worse. Supported representations: `SalientRegionAnnotation`, `SalientRegionPrediction`.
* `salience_f-measure` - f-measure metric for salient object detection task. Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `SalientRegionAnnotation`, `SalientRegionPrediction`.
* `salience_e-measure` - enhanced alignment measure for salient object detection task, defined in following [paper](https://arxiv.org/abs/1805.10421). Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `SalientRegionAnnotation`, `SalientRegionPrediction`.
* `salience_s-measure` - simularity measure for salient object detection task, defined in following [paper](https://arxiv.org/abs/1708.00786). Metric is calculated as a percentage. Direction of metric's growth is higher-better. Supported representations: `SalientRegionAnnotation`, `SalientRegionPrediction`.
* `sisdr` - Scale-invariant signal-to-distortion ratio described in [paper](https://arxiv.org/pdf/1811.02508.pdf). Metric is calculated as a decibel(dB). Direction of metric's mean growth is higher-better. Direction of metric's std growth is higher-worse. Supported representations: `NoiseSuppressionAnnotation`, `NoiseSuppressionPrediction`.
  * `delay` - shift output by delay for alignment with reference (Optional, default 0).
* `normalised_quantile_loss` - q-risk normalized quantile loss for evaluation of time series forecasting models. Direction of metric's growth is higher-worse. Supported representations: `TimeSeriesForecastingAnnotation`, `TimeSeriesForecastingQuantilesPrediction`.
* `log_loss` - classification metric based on probabilities. Direction of metric's growth is higher-worse. Supported representations: `HitRatioAnnotation`, `HitRatioPrediction`.
* `perplexity` - a measurement of how well a probability distribution or probability model predicts a sample. Metric is calculated as a percentage. Direction of metric's growth is higher-worse. Supported representations: `LanguageModelingAnnotation`, `LanguageModelingPrediction`.

## Metrics profiling

Accuracy Checker supports providing detailed information necessary for understanding metric calculation for each data object.
This feature can be useful for debug purposes. For enabling this behaviour you need to provide `--profile True` in accuracy checker command line.
Additionally, you can specify directory for saving profiling results `--profiler_logs_dir` and select data format in `--profile_report_type` between `csv` (brief) and `json` (more detailed).

Supported for profiling metrics:
* Classification:
  * `accuracy`
  * `accuracy_per_class`
  * `classification_f1-score`
  * `metthews_correlation_coef`
  * `multi_accuracy`
  * `multi_recall`
  * `multi_precision`
  * `f1-score`
* Regression:
  * `mae`
  * `mse`
  * `rmse`
  * `mae_on_interval`
  * `mse_on_interval`
  * `rmse_on_interval`
  * `angle_error`
  * `psnr`
  * `ssim`
  * `normed_error`
  * `per_point_normed_error`
* Object Detection:
  * `map`
  * `recall`
  * `miss_rate`
  * `coco_precision`
  * `coco_recall`
* Semantic Segmentation
  * `segmentation_accuracy`
  * `mean_iou`
  * `mean_accuracy`
  * `frequency_weighted_accuracy`
* Instance Segmentation
  * `coco_orig_segm_precision`
* GAN:
  * `inception_score`
  * `fid`
