# Metrics

For correct work metrics require specific representation format.
(e. g. map expects detection annotation and detection prediction for evaluation).

In case when you use complicated representation located in representation container, you need to add options `annotation_source` and `prediction_source` in configuration file to
select specific representation, another way metric calculation possible only if container has only one suitable representation and will be resolved automatically.
`annotation_source` and `prediction_source` should contain only one annotation identifier and output layer name respectively.
You may optionally provide `reference` field for metric, if you want calculated metric tested against specific value (i.e. reported in canonical paper) and acceptable `threshold` for metric deviation from reference value.

Every metric has parameters available for configuration.

Accuracy Checker supports following set of metrics:

* `accuracy` - classification accuracy metric, defined as the number of correct predictions divided by the total number of predictions.
Supported representation: `ClassificationAnnotation`, `TextClassificationAnnotation`, `ClassificationPrediction`.
  * `top_k` - the number of classes with the highest probability, which will be used to decide if prediction is correct.
* `accuracy_per_class` - classification accuracy metric which represents results for each class. Supported representation: `ClassificationAnnotation`, `ClassificationPrediction`.
  * `top_k` - the number of classes with the highest probability, which will be used to decide if prediction is correct.
  * `label_map` - the field in annotation metadata, which contains dataset label map (Optional, should be provided if different from default).
* `character_recognition_accuracy` - accuracy metric for character recognition task. Supported representation: `CharacterRecognitionAnnotation`, `CharacterRecognitionPrediction`.
* `label_level_recognition_accuracy` - [label level recognition accuracy](https://dl.acm.org/doi/abs/10.1145/1143844.1143891) metric for text line character recognition task using [editdistance](https://pypi.org/project/editdistance/). Supported representation: `CharacterRecognitionAnnotation`, `CharacterRecognitionPrediction`.
* `classification_f1-score` - [F1 score](https://en.wikipedia.org/wiki/F1_score) metric for classification task. Supported representation: `ClassificationAnnotation`, `TextClassificationAnnotation`, `ClassificationPrediction`.
* `label_map` - the field in annotation metadata, which contains dataset label map (Optional, should be provided if different from default).
* `metthews_correlation_coef` - [Matthews correlation coefficient (MCC)](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient) for binary classification. Supported representation: `ClassificationAnnotation`, `TextClassificationAnnotation`, `ClassificationPrediction`.
* `map` - mean average precision. Supported representations: `DetectionAnnotation`, `DetectionPrediction`.
  * `overlap_threshold` - minimal value for intersection over union that allows to make decision that prediction bounding box is true positive.
  * `overlap_method` - method for calculation bbox overlap. You can choose between intersection over union (`iou`), defined as area of intersection divided by union of annotation and prediction boxes areas, and intersection over area (`ioa`), defined as area of intersection divided by ara of prediction box.
  * `include_boundaries` - allows include boundaries in overlap calculation process. If it is True then width and height of box is calculated by max - min + 1.
  * `ignore_difficult` - allows to ignore difficult annotation boxes in metric calculation. In this case, difficult boxes are filtered annotations from postprocessing stage.
  * `distinct_conf` - select only values for distinct confidences.
  * `allow_multiple_matches_per_ignored` - allows multiple matches per ignored.
  * `label_map` - the field in annotation metadata, which contains dataset label map  (Optional, should be provided if different from default).
  * `integral` - integral type for average precision calculation. Pascal VOC `11point` and `max` approaches are available.
* `miss_rate` - miss rate metric of detection models.  Supported representations: `DetectionAnnotation`, `DetectionPrediction`.
  * `overlap_threshold` - minimal value for intersection over union that allows to make decision that prediction bounding box is true positive.
  * `overlap_method` - method for calculation bbox overlap. You can choose between intersection over union (`iou`), defined as area of intersection divided by union of annotation and prediction boxes areas, and intersection over area (`ioa`), defined as area of intersection divided by ara of prediction box.
  * `include_boundaries` - allows include boundaries in overlap calculation process. If it is True then width and height of box is calculated by max - min + 1.
  * `ignore_difficult` - allows to ignore difficult annotation boxes in metric calculation. In this case, difficult boxes are filtered annotations from postprocessing stage.
  * `distinct_conf` - select only values for distinct confidences.
  * `allow_multiple_matches_per_ignored` - allows multiple matches per ignored.
  * `label_map` - the field in annotation metadata, which contains dataset label map  (Optional, should be provided if different from default).
  * `fppi_level` - false positive per image level.
* `recall` - recall metric of detection models. Supported representations: `DetectionAnnotation`, `DetectionPrediction`.
  * `overlap_threshold` - minimal value for intersection over union that allows to make decision that prediction bounding box is true positive.
  * `overlap_method` - method for calculation bbox overlap. You can choose between intersection over union (`iou`), defined as area of intersection divided by union of annotation and prediction boxes areas, and intersection over area (`ioa`), defined as area of intersection divided by ara of prediction box.
  * `include_boundaries` - allows include boundaries in overlap calculation process. If it is True then width and height of box is calculated by max - min + 1.
  * `ignore_difficult` - allows to ignore difficult annotation boxes in metric calculation. In this case, difficult boxes are filtered annotations from postprocessing stage.
  * `distinct_conf` - select only values for distinct confidences.
  * `allow_multiple_matches_per_ignored` - allows multiple matches per ignored.
  * `label_map` - the field in annotation metadata, which contains dataset label map (Optional, should be provided if different from default).
* `detection_accuracy` - accuracy for detection models. Supported representations: `DetectionAnnotation`, `DetectionPrediction`.
  * `overlap_threshold` - minimal value for intersection over union that allows to make decision that prediction bounding box is true positive.
  * `overlap_method` - method for calculation bbox overlap. You can choose between intersection over union (`iou`), defined as area of intersection divided by union of annotation and prediction boxes areas, and intersection over area (`ioa`), defined as area of intersection divided by ara of prediction box.
  * `include_boundaries` - allows include boundaries in overlap calculation process. If it is True then width and height of box is calculated by max - min + 1.
  * `label_map` - the field in annotation metadata, which contains dataset label map  (Optional, should be provided if different from default).
  * `use_normalization` - allows to normalize confusion_matrix for metric calculation.
* `segmentation_accuracy` - pixel accuracy for semantic segmentation models. Supported representations: `SegmentationAnnotation`, `SegmentationPrediction`.
  * `use_argmax` - allows to use argmax for prediction mask.
  * `ignore_label` - specified which class_id prediction should be ignored during metric calculation. (Optional, if not provided, all labels will be used)
* `mean_iou` - mean intersection over union for semantic segmentation models. Supported representations: `SegmentationAnnotation`, `SegmentationPrediction`.
  * `use_argmax` - allows to use argmax for prediction mask.
    * `ignore_label` - specified which class_id prediction should be ignored during metric calculation. (Optional, if not provided, all labels will be used)
* `mean_accuracy` - mean accuracy for semantic segmentation models. Supported representations: `SegmentationAnnotation`, `SegmentationPrediction`.
  * `use_argmax` - allows to use argmax for prediction mask.
    * `ignore_label` - specified which class_id prediction should be ignored during metric calculation. (Optional, if not provided, all labels will be used)
* `frequency_weighted_accuracy` - frequency weighted accuracy for semantic segmentation models. Supported representations: `SegmentationAnnotation`, `SegmentationPrediction`.
  * `use_argmax` - allows to use argmax for prediction mask.
  * `ignore_label` - specified which class_id prediction should be ignored during metric calculation. (Optional, if not provided, all labels will be used)
More detailed information about calculation segmentation metrics you can find [here](https://arxiv.org/abs/1411.4038v2).
* `cmc` - Cumulative Matching Characteristics (CMC) score. Supported representations: `ReIdentificationAnnotation`, `ReIdentificationPrediction`.
  * `top_k` -  number of k highest ranked samples to consider when matching.
  * `separate_camera_set` - should identities from the same camera view be filtered out.
  * `single_gallery_shot` -  each identity has only one instance in the gallery.
  * `number_single_shot_repeats` - number of repeats for single_gallery_shot setting (required for CUHK).
  * `first_match_break` - break on first matched gallery sample.
* `reid_map` - Mean Average Precision score for object reidentification. Supported representations: `ReIdentificationAnnotation`, `ReIdentificationPrediction`.
  * `uninterpolated_auc` - should area under precision recall curve be computed using trapezoidal rule or directly.
* `pairwise_accuracy` - pairwise accuracy for object reidentification. Supported representations: `ReIdentificationClassificationAnnotation`, `ReIdentificationPrediction`.
  * `min_score` - min score for determining that objects are different. You can provide value or use `train_median` value which will be calculated if annotations has training subset.
* `pairwise_accuracy_subsets` - object reidentification pairwise accuracy with division dataset on test and train subsets for calculation mean score. Supported representations: `ReIdentificationClassificationAnnotation`, `ReIdentificationPrediction`.
  * `subset_number` - number of subsets for separating.
* `mae` - [Mean Absolute Error](https://en.wikipedia.org/wiki/Mean_absolute_error). Supported representations: `RegressionAnnotation`, `RegressionPrediction`, `FeatureRegressionAnnotation`, `DepthEstimationAnnotation`, `DepthEstimationPrediction`.
* `mae_on_intervals` - Mean Absolute Error estimated magnitude for specific value range. Supported representations: `RegressionAnnotation`, `RegressionPrediction`.
  * `intervals` - comma-separated list of interval boundaries.
  * `ignore_values_not_in_interval` - allows create additional intervals for values less than minimal value in interval and greater than maximal.
  * `start` , `step`, `end` - way to generate range of intervals from `start` to `end` with length `step`.
* `mse` - [Mean Squared Error](https://en.wikipedia.org/wiki/Mean_squared_error). Supported representations: `RegressionAnnotation`, `RegressionPrediction`, `FeatureRegressionAnnotation`, `DepthEstimationAnnotation`, `DepthEstimationPrediction`.
* `mse_on_intervals` - Mean Squared Error estimated magnitude for specific value range. Supported representations: `RegressionAnnotation`, `RegressionPrediction`.
  * `intervals` - comma-separated list of interval boundaries.
  * `ignore_values_not_in_interval` - allows create additional intervals for values less than minimal value in interval and greater than maximal.
  * `start`, `step`, `end` - generate range of intervals from `start` to `end` with length `step`.
* `rmse` - [Root Mean Squared Error](https://en.wikipedia.org/wiki/Root-mean-square_deviation). Supported representations: `RegressionAnnotation`, `RegressionPrediction`, `FeatureRegressionAnnotation`, `DepthEstimationAnnotation`, `DepthEstimationPrediction`.
* `rmse_on_intervals` - Root Mean Squared Error estimated magnitude for specific value range. Supported representations: `RegressionAnnotation`, `RegressionPrediction`.
  * `intervals` - comma-separated list of interval boundaries.
  * `ignore_values_not_in_interval` - allows create additional intervals for values less than minimal value in interval and greater than maximal.
  * `start`, `step`, `end` - generate range of intervals from `start` to `end` with length `step`.
* `per_point_normed_error` - Normed Error for measurement the quality of landmarks' positions. Estimated results for each point independently. Supported representations: `FacialLandmarksAnnotation`, `FacialLandmarksPrediction`, `FacialLandmarks3DAnnotation`, `FacialLandmarks3DPrediction`.
* `normed_error` - Normed Error for measurement the quality of landmarks' positions. Supported representations: `FacialLandmarksAnnotation`, `FacialLandmarksPrediction`, `FacialLandmarks3DAnnotation`, `FacialLandmarks3DPrediction`.
  * `calculate_std` - allows calculation of standard deviation (default value: `False`)
  * `percentile` - calculate error rate for given percentile.
* `nme` - Mean Normed Error for measurement quality of landmarks positions. Supported representations: `FacialLandmarks3DAnnotation`, `FacialLandwarks3DPrediction`.
  `only_2d` - evaluate metric only for 2d case.
* `psnr` - [Peak signal to noise ratio](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio). Supported representations: `SuperResolutionAnnotation`, `SuperResolutionPrediction`, `ImageProcessingAnnotation`, `ImageProcessingPrediction`, `ImageInpaintingAnnotation`, `ImageInpaintingPrediction`.
  * `color_order` - the field specified which color order `BGR` or `RGB` will be used during metric calculation (Optional. Default value is RGB), used only if you have 3-channel images.
* `ssim` - [Structural similarity](https://en.wikipedia.org/wiki/Structural_similarity). Supported representations: `ImageProcessingAnnotation`, `ImageProcessingPrediction`, `ImageInpaintingAnnotation`, `ImageInpaintingPrediction`, `SuperResolutionAnnotation`, `SuperResolutionPrediction`.
* `angle_error` - Mean angle error and Standard deviation of angle error for gaze estimation. Supported representations: `GazeVectorAnnotation`, `GazeVectorPrediction`.
* `multi_accuracy` - accuracy for multilabel recognition task. Supported representations: `MultiLabelRecognitionAnnotation`, `MultiLabelRecognitionPrediction`.
  * `label_map` - the field in annotation metadata, which contains dataset label map (Optional, should be provided if different from default).
  * `calculate_average` - allows calculation of average accuracy (default value: `True`).
* `multi_precision` - precision metric for multilabel recognition. Supported representations: `MultiLabelRecognitionAnnotation`, `MultiLabelRecognitionPrediction`.
  * `label_map` - the field in annotation metadata, which contains dataset label map  (Optional, should be provided if different from default).
  * `calculate_average` - allows calculation of average precision (default value: `True`).
* `multi_recall` - recall metric for multilabel recognition. Supported representations: `MultiLabelRecognitionAnnotation`, `MultiLabelRecognitionPrediction`.
  * `label_map` - the field in annotation metadata, which contains dataset label map  (Optional, should be provided if different from default).
  * `calculate_average` - allows calculation of average recall (default value: `True`).
* `f1_score` - [F score](https://en.wikipedia.org/wiki/F1_score) metric for multilabel recognition. Supported representations: `MultiLabelRecognitionAnnotation`, `MultiLabelRecognitionPrediction`.
  * `label_map` - the field in annotation metadata, which contains dataset label map  (Optional, should be provided if different from default).
  * `calculate_average` - allows calculation of average f-score (default value: `True`).
* `focused_text_hmean` - Harmonic mean of precision and recall for focused scene text detection task introduced in [Robust Reading Competition challenge 2](https://rrc.cvc.uab.es/?ch=2&com=introduction). Supported representations: `TextDetectionAnnotation`, `TextDetectionPrediction`, `DetectionPrediction`.
  * `ignore_difficult` - allows to ignore difficult ground truth text polygons in metric calculation.
  * `area_precision_constrain` - minimal value for precision that allows to make decision that prediction polygon matched with annotation.
  * `area_recall_constrain` - minimal value for recall that allows to make decision that prediction polygon matched with annotation.
  * `center_diff_threshold` - acceptable difference between center of predicted text region and ground truth.
  * `one_to_one_match_score` - weight for one to one matching results (Optional, default 1).
  * `one_to_many_match_score` - weight for one to many matching results (Optional, default 0.8).
  * `many_to_one_match_score` - weight for many to one matching results (optional, default 1).
* `focused_text_precision` - precision for focused scene text detection task introduced in [Robust Reading Competition challenge 2](https://rrc.cvc.uab.es/?ch=2&com=introduction). Supported representations: `TextDetectionAnnotation`, `TextDetectionPrediction`, `DetectionPrediction`.
  * `ignore_difficult` - allows to ignore difficult ground truth text polygons in metric calculation.
  * `area_precision_constrain` - minimal value for precision that allows to make decision that prediction polygon matched with annotation.
  * `area_recall_constrain` - minimal value for recall that allows to make decision that prediction polygon matched with annotation.
  * `center_diff_threshold` - acceptable difference between center of predicted text region and ground truth.
  * `one_to_one_match_score` - weight for one to one matching results (Optional, default 1).
  * `one_to_many_match_score` - weight for one to many matching results (Optional, default 0.8).
  * `many_to_one_match_score` - weight for many to one matching results (optional, default 1).
* `focused_text_precision` - recall for focused scene text detection task introduced in [Robust Reading Competition challenge 2](https://rrc.cvc.uab.es/?ch=2&com=introduction). Supported representations: `TextDetectionAnnotation`, `TextDetectionPrediction`, `DetectionPrediction`.
  * `ignore_difficult` - allows to ignore difficult ground truth text polygons in metric calculation.
  * `area_precision_constrain` - minimal value for precision that allows to make decision that prediction polygon matched with annotation.
  * `area_recall_constrain` - minimal value for recall that allows to make decision that prediction polygon matched with annotation.
  * `center_diff_threshold` - acceptable difference between center of predicted text region and ground truth.
  * `one_to_one_match_score` - weight for one to one matching results (Optional, default 1).
  * `one_to_many_match_score` - weight for one to many matching results (Optional, default 0.8).
  * `many_to_one_match_score` - weight for many to one matching results (optional, default 1).
* `incidental_text_hmean` - Harmonic mean of precision and recall for incidental scene text detection task introduced in [Robust Reading Competition challenge 4](https://rrc.cvc.uab.es/?ch=4&com=introduction). Supported representations: `TextDetectionAnnotation`, `TextDetectionPrediction`.
  * `ignore_difficult` - allows to ignore difficult ground truth text polygons in metric calculation.
  * `iou_constrain` - minimal value for intersection over union that allows to make decision that prediction polygon is true positive.
  * `area_precision_constrain` - minimal value for precision that allows to make decision that prediction polygon matched with ignored annotation.
* `incidental_text_precision` - precision for incidental scene text detection task introduced in [Robust Reading Competition challenge 4](https://rrc.cvc.uab.es/?ch=4&com=introduction). Supported representations: `TextDetectionAnnotation`, `TextDetectionPrediction`.
  * `ignore_difficult` - allows to ignore difficult ground truth text polygons in metric calculation.
  * `iou_constrain` - minimal value for intersection over union that allows to make decision that prediction polygon is true positive.
  * `area_precision_constrain` - minimal value for precision that allows to make decision that prediction polygon matched with ignored annotation.
* `incidental_text_precision` - recall for incidental scene text detection task introduced in [Robust Reading Competition challenge 4](https://rrc.cvc.uab.es/?ch=4&com=introduction). Supported representations: `TextDetectionAnnotation`, `TextDetectionPrediction`.
  * `ignore_difficult` - allows to ignore difficult ground truth text polygons in metric calculation.
  * `iou_constrain` - minimal value for intersection over union that allows to make decision that prediction polygon is true positive.
  * `area_precision_constrain` - minimal value for precision that allows to make decision that prediction polygon matched with ignored annotation.
* `coco_precision` - MS COCO Average Precision metric for keypoints recognition and object detection tasks. Supported representations: `PoseEstimationAnnotation`, `PoseEstimationPrediction`, `DetectionAnnotation`, `DetectionPrediction`.
  * `max_detections` - max number of predicted results per image. If you have more predictions,the results with minimal confidence will be ignored.
  * `threshold` - intersection over union threshold. You can specify one value or comma separated range of values. This parameter supports precomputed values for standard COCO thresholds (`.5`, `.75`, `.5:.05:.95`).
* `coco_recall` - MS COCO Average Recall metric for keypoints recognition and object detection tasks. Supported representations: `PoseEstimationAnnotation`, `PoseEstimationPrediction`, `DetectionAnnotation`, `DetectionPrediction`.
  * `max_detections` - max number of predicted results per image. If you have more predictions,the results with minimal confidence will be ignored.
  * `threshold` - intersection over union threshold. You can specify one value or comma separated range of values. This parameter supports precomputed values for standard COCO thresholds (`.5`, `.75`, `.5:.05:.95`).
* `coco_keypoints_precision` - MS COCO Average Precision metric for keypoints recognition task. Supported representations: `PoseEstimationAnnotation`, `PoseEstimationPrediction`.
  * `max_detections` - max number of predicted results per image. If you have more predictions,the results with minimal confidence will be ignored.
  * `threshold` - intersection over union threshold. You can specify one value or comma separated range of values. This parameter supports precomputed values for standard COCO thresholds (`.5`, `.75`, `.5:.05:.95`).
* `coco_keypoints_recall` - MS COCO Average Recall metric for keypoints recognition task. Supported representations: `PoseEstimationAnnotation`, `PoseEstimationPrediction`.
  * `max_detections` - max number of predicted results per image. If you have more predictions,the results with minimal confidence will be ignored.
  * `threshold` - intersection over union threshold. You can specify one value or comma separated range of values. This parameter supports precomputed values for standard COCO thresholds (`.5`, `.75`, `.5:.05:.95`).
* `hit_ratio` - metric for recommendation system evaluation. Supported representations: `HitRatioAnnotation`, `HitRatioPrediction`.
  * `top_k` - definition of number elements in rank list (optional, default 10).
* `ndcg` - [Normalized Discounted Cumulative Gain](https://en.wikipedia.org/wiki/Discounted_cumulative_gain). Supported representations: `HitRatioAnnotation`, `HitRatioPrediction`.
  * `top_k` - definition of number elements in rank list (optional, default 10).
* `dice` - [Sørensen–Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient). Supported representations: `BrainTumorSegmentationAnnotation, BrainTumorSegmentationPrediction`.
* `dice_index` - [Sørensen–Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient). Supported representations: `BrainTumorSegmentationAnnotation, BrainTumorSegmentationPrediction`, `SegmentationAnnotation, SegmentationPrediction`. Supports result representation for multiple classes. Metric represents result for each class if `label_map` for used dataset is provided, otherwise it represents overall result. For `brats_numpy` converter file with labels set in `labels_file` tag.
  * `mean` - allows calculation mean value (default - `True`).
  * `median` - allows calculation median value (default - `False`).
* `bleu` - [Bilingual Evaluation Understudy](https://en.wikipedia.org/wiki/BLEU). Supperted representations: `MachineTranslationAnnotation`, `MachineTranslationPrediction`.
  * `smooth` - Whether or not to apply Lin et al. 2004 smoothing.
  *  `max_order` - Maximum n-gram order to use when computing BLEU score. (Optional, default 4).
* `f1` - F1-score for question answering task. Supported representations: `QuestionAnsweringAnnotation`, `QuestionAnsweringPrediction`.
* `exact_match` - Exact matching (EM) metric for question answering task. Supported representations: `QuestionAnsweringAnnotation`, `QuestionAnsweringPrediction`.
* `mpjpe_multiperson` - [Mean Per Joint Position Error](http://vision.imar.ro/human3.6m/pami-h36m.pdf) extended for multi-person case. Supported representations: `PoseEstimation3dAnnotation`, `PoseEstimation3dPrediction`. As the first step, correspondence between ground truth and prediction skeletons is established for each image. Then MPJPE is computed for each ground truth and prediction pair. The error is averaged over poses in each frame, then over all frames.
* `face_recognition_tafa_pair_metric` - accuracy for face recognition models based on dot product of embedding values. Supported representations: `ReIdentificationAnnotation`, `ReIdentificationPrediction`.
  * `threshold` - minimal dot product value of embeddings to identify as matched face pair.
* `youtube_faces_accuracy` - accuracy for face detection models calculated based on IOU values of ground truth bounding boxes and model-detected bounding boxes. Supported representations: `DetectionAnnotation`, `DetectionPrediction`.
  * `overlap` - minimum IOU threshold to consider as a true positive candidate face.
  * `relative_size` - size of detected face candidate\'s area in proportion to the size of ground truth\'s face size. This value is set to filter candidates that have high IOU but have a relatively smaller face size than ground truth face size.
* `normalized_embedding_accuracy` - accuracy for re-identification models based on normalized dot product of embedding values. Supported representations: `ReIdentificationAnnotation`, `ReIdentificationPrediction`.
  * `top_k` - the number of people with the highest probability, which will be used to decide if prediction is correct.
* `attribute_accuracy` - accuracy for attributes in attribute classification models. Supported representations: `ContainerAnnotation` with `ClassificationAnnotation` and `ContainerPrediction` with `ClassificationPrediction`.
  * `attributes`: names of attributes.
  * `calculate_average` - allows calculation of average accuracy (default value: `True`).
* `attribute_recall` - recall for attributes in attribute classification models. Supported representations: `ContainerAnnotation` with `ClassificationAnnotation` and `ContainerPrediction` with `ClassificationPrediction`.
  * `attributes`: names of attributes.
  * `calculate_average` - allows calculation of average recall (default value: `True`).
* `attribute_precision` - precision for attributes in attribute classification models. Supported representations: `ContainerAnnotation` with `ClassificationAnnotation` and `ContainerPrediction` with `ClassificationPrediction`.
  * `attributes`: names of attributes.
  * `calculate_average` - allows calculation of average precision (default value: `True`).
* `wer` - Word error rate ([WER](https://en.wikipedia.org/wiki/Word_error_rate)). Supported representations: `CharacterRecognitionAnnotation`, `CharacterRecognitionPrediction`.
* `greedy_wer` - approach to calculate WER as length normalized [edit distance](https://en.wikipedia.org/wiki/Edit_distance). Supported representations: `CharacterRecognitionAnnotation`, `CharacterRecognitionPrediction`.
* `score_class_comparison` - allows calculate an accuracy of quality score class(low/normal/good). It sorts all quality scores from annotations and predictions and set the k1 highest scores as high class and the k2 lowest scores as low class where k1 is `num_high_quality` and k2 is `num_low_quality`. Supported representations: `QualityAssessmentAnnotation`, `QualityAssessmentPrediction`.
  * `num_high_quality` - the number of high class in total (default value: `1`).
  * `num_low_quality` - the number of low class in total (default value: `1`).
