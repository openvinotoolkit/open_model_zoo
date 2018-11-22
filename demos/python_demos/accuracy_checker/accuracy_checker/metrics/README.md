# Metrics

For correct work metrics require specific representation format. 
(e. g. map expects detection annotation and detection prediction for evaluation). 

In case when you use complicated representation located in representation container, you need to add options `annotation_source` and `prediction_source` in configuration file to
select specific representation, another way metric calculation possible only if conteiner has only one suitable representation and will be resolved automatically. 
`annotation_source` and `prediction_source` should contain only one annotation identifier and output layer name respectively.
You may optionally provide `reference` field for metric, if you want calculated metric tested against specific value (i.e. reported in canonical paper) and acceptable `threshold` for metric deviation from reference value.

Every metric has parameters available for configuration. 

Accuracy Checker supports following set of metrics:

* `accuracy` - classification accuracy metric, defined as the number of correct predictions divided by the total number of predictions.
Supported reprsentation: `ClassificationAnnotation`, `ClassificationPrediction`
  * `top_k` - the number of classes with the highest probability, which will be used to decide if prediction is correct.
* `accuracy_per_class` - classification accuracy metric which represents results for each class. Supported reprsentation: `ClassificationAnnotation`, `ClassificationPrediction`.
  * `top_k` - the number of classes with the highest probability, which will be used to decide if prediction is correct.
  * `label_map` - the field in annotation metadata, which contains dataset label map.
* `character_recognition_accuracy` - accuracy metric for character recognition task. Supported reprsentation: `CharacterRecognitionAnnotation`, `CharacterRecognitionPrediction`.
* `map` - mean average precision. Supported representations: `DetectionAnnotation`, `DetectionPrediction`.
  * `overlap_threshold` - minimal value for intersection over union that allows to make decision that prediction bounding box is true positive.
  * `overlap_method` - method for calculation bbox overlap. You can choose between intersection over union (`iou`), defined as area of intersection devied by union of annoation and prediction boxes areas, and intersection over area (`ioa`), defined as area of intersection devied by ara of prediction box.
  * `include_boundaries` - allows include boundaries in overlap calculation process. If it is True then width and height of box is calculated by max - min + 1.
  * `ignore_difficult` - allows to ignore difficult aanotation boxes in metric calculation. In this case, difficult boxes are filtered annotations from postprocessing stage.
  * `distinct_conf` - select only values for distinct confidences.
  * `allow_multiple_matches_per_ignored` - allows multiple matches per ignored.
  * `label_map` - the field in annotation metadata, which contains dataset label map.
  * `integral` - integral type for average precesion calculation. Pascal VOC `11point` and `max` approaches are available.
* `miss_rate` - miss rate metric of detection models.  Supported representations: `DetectionAnnotation`, `DetectionPrediction`.
  * `overlap_threshold` - minimal value for intersection over union that allows to make decision that prediction bounding box is true positive.
  * `overlap_method` - method for calculation bbox overlap. You can choose between intersection over union (`iou`), defined as area of intersection devied by union of annoation and prediction boxes areas, and intersection over area (`ioa`), defined as area of intersection devied by ara of prediction box.
  * `include_boundaries` - allows include boundaries in overlap calculation process. If it is True then width and height of box is calculated by max - min + 1.
  * `ignore_difficult` - allows to ignore difficult aanotation boxes in metric calculation. In this case, difficult boxes are filtered annotations from postprocessing stage.
  * `distinct_conf` - select only values for distinct confidences.
  * `allow_multiple_matches_per_ignored` - allows multiple matches per ignored.
  * `label_map` - the field in annotation metadata, which contains dataset label map.
  * `fppi_level` - false positive per image lavel.
* `recall` - recall metric of detection models. Supported representations: `DetectionAnnotation`, `DetectionPrediction`.
  * `overlap_threshold` - minimal value for intersection over union that allows to make decision that prediction bounding box is true positive.
  * `overlap_method` - method for calculation bbox overlap. You can choose between intersection over union (`iou`), defined as area of intersection devied by union of annoation and prediction boxes areas, and intersection over area (`ioa`), defined as area of intersection devied by ara of prediction box.
  * `include_boundaries` - allows include boundaries in overlap calculation process. If it is True then width and height of box is calculated by max - min + 1.
  * `ignore_difficult` - allows to ignore difficult aanotation boxes in metric calculation. In this case, difficult boxes are filtered annotations from postprocessing stage.
  * `distinct_conf` - select only values for distinct confidences.
  * `allow_multiple_matches_per_ignored` - allows multiple matches per ignored.
  * `label_map` - the field in annotation metadata, which contains dataset label map.
* `segmentation_accuracy` - pixel accuracy for semantic segmentation models. Supported representations: `SegmentationAnnotation`, `SegmentationPrediction`.
* `mean_iou` - mean intersection over union for semantic segmentation models. Supported representations: `SegmentationAnnotation`, `SegmentationPrediction`.
* `mean_accuracy` - mean accuracy for semantic segmentation models. Supported representations: `SegmentationAnnotation`, `SegmentationPrediction`.
* `frequency_weighted_accuracy` - freaquency weighted accuracy for semantic segmentation models. Supported representations: `SegmentationAnnotation`, `SegmentationPrediction`.
More detailed information about calculation segmentation metrics you can find [here][segmentation_article].
* `cmc` - Cumulative Matching Characteristics (CMC) score. Supported representations: `ReIdentificationAnnotation`, `ReIdentificationPrediction`.
  * `top_k` -  number of k highest ranked samples to consider when matching.
  * `separate_camera_set` - should identities from the same camera view be filtered out.
  * `single_gallery_shot` -  each identity has only one instance in the gallery.
  * `number_single_shot_repeats` - number of repeats for single_gallery_shot setting (required for CUHK).
  * `first_match_break` - break on first matched gallery sample.
* `reid_map` - Mean Average Precision score for object reidentificatin. Supported representations: `ReIdentificationAnnotation`, `ReIdentificationPrediction`.
  * `uninterpolated_auc` - should area under precision recall curve be computed using trapezoidal rule or directly.
*  `pairwise_accuracy` - pairwise accuracy for object reidentification. Supported representations: `ReIdentificationClassificationAnnotation`, `ReIdentificationPrediction`.
  * `min_score` - min score for determining that objects are different. You can provide value or use `train_median` value which will be calculated if annotations has training subset.
* `pairwise_accuracy_subsets` - object reidentification pairwise accuracy with devision dataset on test and train sabsets for calculation mean score. Supported representations: `ReIdentificationClassificationAnnotation`, `ReIdentificationPrediction`.
  * `subset_number` - number of subsets for separating. 
* `mae` - [Mean Absolute Error][mae]. Supported representations: `RegressionAnnotation`, `RegressionPrediction`.
* `mae_on_intervals` - Mean Absolute Error estimated magnitude for specific value range. Supported representations: `RegressionAnnotation`, `RegressionPrediction`.
  * `intervals` - comma-separated list of interval boundaries.
  * `ignore_values_not_in_interval` - allows create additional intervals for values less than minimal value in interval and greater than maximal.
  * `start` , `step`, `end` - way to generate range of intervals from `start` to `end` with lenght `step`.
* `mse` - [Mean Squared Error][mse]. Supported representations: `RegressionAnnotation`, `RegressionPrediction`.
* `mse_on_intervals` - Mean Squared Error estimated magnitude for specific value range. Supported representations: `RegressionAnnotation`, `RegressionPrediction`.
  * `intervals` - comma-separated list of interval boundaries.
  * `ignore_values_not_in_interval` - allows create additional intervals for values less than minimal value in interval and greater than maximal.
  * `start`, `step`, `end` - generate range of intervals from `start` to `end` with lenght `step`.
* `rmse` - [Root Mean Squared Error][rmse]. Supported representations: `RegressionAnnotation`, `RegressionPrediction`.
* `rmse_on_intervals` - Root Mean Squared Error estimated magnitude for specific value range. Supported representations: `RegressionAnnotation`, `RegressionPrediction`.
  * `intervals` - comma-separated list of interval boundaries.
  * `ignore_values_not_in_interval` - allows create additional intervals for values less than minimal value in interval and greater than maximal.
  * `start`, `step`, `end` - generate range of intervals from `start` to `end` with lenght `step`.
* `per_point_regression` - Root Mean Squared Error for 2D points estimated results for each point independently. Supported representations: `PointRegressionAnnotation`, `PointRegressionPrediction`.
  * `scaling_distance` - comma-separited list of 2 point indexes, distance between which will be used for scaling regression distances.
* `average point error` - Root Mean Squared Error for 2D points estimated average results for all points. Supported representations: `PointRegressionAnnotation`, `PointRegressionPrediction`.
  * `scaling_distance` - comma-separited list of 2 point indexes, distance between which will be used for scaling regression distances.
* `multi_accuracy` - accuracy for multilabel recognition task. Supported representations: `MultilabelRecognitionAnnotation`, `MultilabelRecognitionPrediction`.
  * `label_map` - the field in annotation metadata, which contains dataset label map.
* 'multi_precision' - precision metric for multilabel recognition. Supported representations: `MultilabelRecognitionAnnotation`, `MultilabelRecognitionPrediction`.
  * `label_map` - the field in annotation metadata, which contains dataset label map.
* `multi_recall` - recall metric for multilabel recognition. Supported representations: `MultilabelRecognitionAnnotation`, `MultilabelRecognitionPrediction`.
  * `label_map` - the field in annotation metadata, which contains dataset label map.
* `f1_score` - [F score][f_score] metric for multilabel recognition. Supported representations: `MultilabelRecognitionAnnotation`, `MultilabelRecognitionPrediction`.
  * `label_map` - the field in annotation metadata, which contains dataset label map.

[segmentation_article]: https://arxiv.org/pdf/1411.4038v2.pdf
[mae]: https://en.wikipedia.org/wiki/Mean_absolute_error
[mse]: https://en.wikipedia.org/wiki/Mean_squared_error
[rmse]: https://en.wikipedia.org/wiki/Root-mean-square_deviation
[f_score]: https://en.wikipedia.org/wiki/F1_score
