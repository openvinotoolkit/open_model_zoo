# Adapters

Adapter is a class for converting raw network infer output to specific representation format which is suitable for the further postprocessors work and the metrics calculation. Adapters may have parameters available for configuration. The adapter and its parameters, if necessary, are set through the configuration file.

## Describing how to set adapter in Configuration File

Adapters can be provided in `launchers` section of configuration file for each launcher to use specific adapter.

You can use 2 ways to set adapter for topology:
* Define adapter as a string.

```yml
adapter: classification
```

* Define adapter as a dictionary, using `type:` for setting adapter name. This approach gives opportunity to set additional parameters for adapter if it is required.

```yml
adapter:
  type: reid
  grn_workaround: False
```

## Supported Adapters

AccuracyChecker supports following set of adapters:
* `classification` - converting output of classification model to `ClassificationPrediction` representation.
  * `argmax_output` - identifier that model output is ArgMax layer.
  * `block` - process whole batch as a single data block.
  * `classification_output` - target output layer name.
* `segmentation` - converting output of semantic segmentation model to `SeegmentationPrediction` representation.
  * `make_argmax` - allows applying argmax operation to output values.
* `segmentation_one_class` - converting output of semantic segmentation to `SeegmentationPrediction` representation. It is suitable for situation when model's output is probability of belong each pixel to foreground class.
  * `threshold` - minimum probability threshold for valid class belonging.
* `tiny_yolo_v1` - converting output of Tiny YOLO v1 model to `DetectionPrediction` representation.
* `reid` - converting output of reidentification model to `ReIdentificationPrediction` representation.
  * `grn_workaround` - enabling processing output with adding Global Region Normalization layer.
  * `joining_method` - method used to join embeddings (optional, supported methods are `sum` and `concatenation`, default - `sum`).
* `yolo_v2` - converting output of YOLO v2 family models to `DetectionPrediction` representation.
  * `classes` - number of detection classes (default 20).
  * `anchors` - anchor values provided as comma-separated list or one of precomputed:
    - `yolo_v2` - `[1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]`,
    - `tiny_yolo_v2` - `[1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]`
  * `coords` - number of bbox coordinates (default 4).
  * `num` - num parameter from DarkNet configuration file (default 5).
  * `cells` - number of cells across width and height (default 13).
  * `raw_output` - enabling additional preprocessing for raw YOLO output format (default `False`).
  * `output_format` - setting output layer format:
      - `BHW` - boxes first (default, also default for generated IRs).
      - `HWB` - boxes last.
      Applicable only if network output not 3D (4D with batch) tensor.
* `yolo_v3` - converting output of YOLO v3 family models to `DetectionPrediction` representation.
  * `classes` - number of detection classes (default 80).
  * `anchors` - anchor values provided as comma-separated list or precomputed:
    - `yolo_v3` - `[10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0]`
    - `tiny_yolo_v3` - `[10.0, 14.0, 23.0, 27.0, 37.0, 58.0, 81.0, 82.0, 135.0, 169.0, 344.0, 319.0]`
  * `coords` - number of bbox coordinates (default 4).
  * `num` - num parameter from DarkNet configuration file (default 3).
  * `anchor_mask` - mask for used anchors for each output layer (Optional, if not provided default way for selecting anchors will be used.)
  * `threshold` - minimal objectness score value for valid detections (default 0.001).
  * `outputs` - the list of output layers names.
  * `raw_output` - enabling additional preprocessing for raw YOLO output format (default `False`).
  * `output_format` - setting output layer format - boxes first (`BHW`)(default, also default for generated IRs), boxes last (`HWB`). Applicable only if network output not 3D (4D with batch) tensor.
  * `cells` - sets grid size for each layer, according `outputs` filed. Works only with `do_reshape=True` or when output tensor dimensions not equal 3.
  * `do_reshape` - forces reshape output tensor to [B,Cy,Cx] or [Cy,Cx,B] format, depending on `output_format` value ([B,Cy,Cx] by default). You may need to specify `cells` value.
  * `transpose` - transpose output tensor to specified format (optional).
* `yolo_v3_onnx` - converting output of ONNX Yolo V3 model to `DetectionPrediction`.
  * `boxes_out` - the name of layer with bounding boxes
  * `scores_out` - the name of output layer with detection scores for each class and box pair.
  * `indices_out` - the name of output layer with indices triplets (class_id, score_id, bbox_id).
* `yolo_v3_tf2` - converting output of TensorFlow 2 Yolo V3 with embedded box decoding to `DetectionPrediction`.
  * `outputs` - the list of output layers names.
  * `score_threshold` - minimal accepted score for valid boxes (Optional, default 0).
* `yolo_v5` - converting output of YOLO v5 family models to `DetectionPrediction` representation. The parameters are the same as for the `yolo_v3` models.
* `lpr` - converting output of license plate recognition model to `CharacterRecognitionPrediction` representation.
* `aocr` - converting output of attention-ocr model to `CharacterRecognitionPrediction`.
  * `output_blob` - name of output layer with predicted labels or string (Optional, if not provided, first founded output will be used).
  * `labels` - optional, list of supported tokens for decoding raw labels (Optional, default configuration is ascii charmap, this parameter ignored if you have decoding part in the model).
  * `eos_index` - index of end of string token in labels. (Optional, default 2, ignored if you have decoding part in the model).
  * `to_lower_case` - allow converting decoded characters to lower case (Optional, default is `True`).
* `ppocr` - converting PaddlePaddle CRNN-like model output to `CharacterRecognitionPrediction`.
  * `vocabulary_file` - file with recognition symbols for decoding.
  * `remove_duplicates` - allow removing of duplicated symbols (Optional, default value - `True`).
* `ssd` - converting  output of SSD model to `DetectionPrediction` representation.
* `ssd_mxnet` - converting output of SSD-based models from MXNet framework to `DetectionPrediction` representation.
* `pytorch_ssd_decoder` - converts output of SSD model from PyTorch without embedded decoder.
  * `scores_out` - name of output layer with bounding boxes scores.
  * `boxes_out` - name of output layer with bounding boxes coordinates.
  * `confidence_threshold` - lower bound for valid boxes scores (optional, default 0.05).
  * `nms_threshold` - overlap threshold for NMS (optional, default 0.5).
  * `keep_top_k ` - maximal number of boxes which should be kept (optional, default 200).
  * `feat_size` - features size in format [[feature_width, feature_height], ...] (Optional, default values got from [MLPerf](https://github.com/mlcommons/inference/blob/0691366473fd4fbbc4eb432fad990683a5a87099/v0.5/classification_and_detection/python/models/ssd_r34.py#L208))
  * `do_softmax` - boolean flag which says should be softmax applied to detection scores or not. (Optional, default True)
* `ssd_onnx` - converting output of SSD-based model from PyTorch with NonMaxSuppression layer.
  * `labels_out` - name of output layer with labels or regular expression for it searching.
  * `scores_out`- name of output layer with scores or regular expression for it searching. Optional, can be not provided, if your model has concatenation of scores with box coordinates.
  * `bboxes_out` - name of output layer with bboxes or regular expression for it searching.
* `tf_object_detection` - converting output of detection models from TensorFlow object detection API to `DetectionPrediction`.
  * `classes_out` - name of output layer with predicted classes.
  * `boxes_out` - name of output layer with predicted boxes coordinates in format [y0, x0, y1, x1].
  *  `scores_out` - name of output layer with detection scores.
  * `num_detections_out` - name of output layer which contains the number of valid detections.
* `faster_rcnn_onnx` - converts output of ONNX Faster RCNN model to `DetectionPrediction`
  * `labels_out` - name of output layer with labels, optional if labels concatenated with boxes and scores (only boxes output provided and it has shape [N, 6]).
  * `scores_out`- name of output layer with scores, optional if scores concatenated with boxes (boxes output has shape [N, 5]).
  * `bboxes_out` - name of output layer with bboxes.
* `retinanet` - converting output of RetinaNet-based model.
  * `loc_out` - name of output layer with bounding box deltas.
  * `class_out` - name of output layer with classification probabilities.
* `retinanet_multihead` - converting output of RetinaNet model with multiple level outputs.
  * `boxes_outputs` - list of outputs with boxes.
  * `class_outputs` - list of outputs with class probabilities.
  **Important note: the number of boxes outputs and class outputs should be equal.**
  * `ratios` - the list of ratios for anchor generation (Optional, default [1.0, 2.0, 0.5]).
  * `pre_nms_top_k` - keep top k boxes before NMS applied (Optional, default 1000).
  * `post_nms_top_k` - final number of detections after NMS applied (Optional, default 100).
  * `nms_threshold` - threshold for NMS (Optional, default 0.5).
  * `min_conf` - minimal confidence threshold for detections (Optional, default 0.05).
* `retinanet_tf2` - converting output of RetinaNet-based model from TensorFlow 2 official implementation.
  * `boxes_outputs` - list of outputs with boxes.
  * `class_outputs` - list of outputs with class probabilities.
  **Important note: the number of boxes outputs and class outputs should be equal.**
  * `aspect_ratios` - the list of aspect ratios for anchor generation (Optional, default [1.0, 2.0, 0.5]).
  * `min_level` - minimal pyramid level (Optional, default 3).
  * `max_level` - maximal pyramid level (Optional, default 7).
  * `num_scales` - number of anchor scales (Optional, default 3).
  * `anchor_size` - size of anchor box (Optional, default 4).
  * `pre_nms_top_k` - keep top k boxes before NMS applied (Optional, default 5000).
  * `total_size` - final number of detections after NMS applied (Optional, default 100).
  * `nms_threshold` - threshold for NMS (Optional, default 0.5).
  * `score_threshold` - minimal confidence threshold for detections (Optional, default 0.05).
* `rfcn_class_agnostic` - convert output of Caffe RFCN model with agnostic bounding box regression approach.
  * `cls_out` - the name of output layer with detected probabilities for each class. The layer shape is [num_boxes, num_classes], where `num_boxes` is number of predicted boxes, `num_classes` - number of classes in the dataset including background.
  * `bbox_out` - the name of output layer with detected boxes deltas. The layer shape is [num_boxes, 8] where  `num_boxes` is number of predicted boxes, 8 (4 for background + 4 for foreground) bounding boxes coordinates.
  * `roid_out` - the name of output layer with regions of interest.
* `face_person_detection` - converting face person detection model output with 2 detection outputs to `ContainerPredition`, where value of parameters `face_out`and `person_out` are used for identification `DetectionPrediction` in container.
  * `face_out` -  face detection output layer name.
  * `person_out` - person detection output layer name.
* `person_attributes` - converting person attributes recognition model output to `MultiLabelRecognitionPrediction`.
  * `attributes_recognition_out` - output layer name with attributes scores. (optional, used if your model has more than one outputs).
* `vehicle_attributes`  - converting vehicle attributes recognition model output to `ContainerPrediction` where value of parameters `color_out`and `type_out` are used for identification `ClassificationPrediction` in container.
  * `color_out` - vehicle color attribute output layer name.
  * `type_out`- vehicle type attribute output layer name.
* `head_pose` - converting head pose estimation model output to `ContainerPrediction` where names of parameters `angle_pitch`, `angle_yaw` and `angle_roll` are used for identification `RegressionPrediction` in container.
  * `angle_pitch` - output layer name for pitch angle.
  * `angle_yaw`- output layer name for yaw angle.
  * `angle_roll` - output layer name for roll angle.
* `age_gender` - converting age gender recognition model output to `ContainerPrediction` with `ClassificationPrediction` named `gender` for gender recognition, `ClassificationPrediction` named `age_classification` and `RegressionPrediction` named `age_error` for age recognition.
  * `age_out` - output layer name for age recognition.
  * `gender_out` - output layer name for gender recognition.
* `age_recognition` - converting age recognition model output to `ContainerPrediction` with `ClassificationPrediction` named `age_classification` and `RegressionPrediction` named `age_error` for age recognition.
  * `age_out` - output layer name for age recognition (Optional).
* `action_detection` - converting output of model for person detection and action recognition tasks to `ContainerPrediction` with `DetectionPrediction` for class agnostic metric calculation and `ActionDetectionPrediction` for action recognition. The representations in container have names `class_agnostic_prediction` and `action_prediction` respectively.
  * `priorbox_out` - name of layer containing prior boxes in SSD format.
  * `loc_out` - name of layer containing box coordinates in SSD format.
  * `main_conf_out` - name of layer containing detection confidences.
  * `add_conf_out_prefix` - prefix for generation name of layers containing action confidences if topology has several following layers or layer name.
  * `add_conf_out_count` - number of layers with action confidences (optional, you can not provide this argument if action confidences contained in one layer).
  * `num_action_classes` - number classes for action recognition.
  * `detection_threshold` - minimal detection confidences level for valid detections.
  * `actions_scores_threshold` - minimal actions confidences level for valid detections.
  * `action_scale` - scale for correct action score calculation.
* `image_processing` - converting output of network for single image processing to `ImageProcessingPrediction`.
  * `reverse_channels` - allow switching output image channels e.g. RGB to BGR (Optional. Default value is False).
  * `mean` - value or list channel-wise values which should be added to result for getting values in range [0, 255] (Optional, default 0)
  * `std` - value or list channel-wise values on which result should be multiplied for getting values in range [0, 255] (Optional, default 255)
  **Important** Usually `mean` and `std` are the same which used in preprocessing, here they are used for reverting these preprocessing operations.
  The order of actions:
  1. Multiply on `std`
  2. Add `mean`
  3. Reverse channels if this option enabled.
  * `target_out` - target model output layer name in case when model has several outputs.
* `super_resolution` - converting output of single image super resolution network to `SuperResolutionPrediction`.
  * `reverse_channels` - allow switching output image channels e.g. RGB to BGR (Optional. Default value is False).
  * `mean` - value or list channel-wise values which should be added to result for getting values in range [0, 255] (Optional, default 0)
  * `std` - value or list channel-wise values on which result should be multiplied for getting values in range [0, 255] (Optional, default 255)
  * `cast_to_uint8` - perform casting output image pixels to [0, 255] range.
  **Important** Usually `mean` and `std` are the same which used in preprocessing, here they are used for reverting these preprocessing operations.
  The order of actions:
  1. Multiply on `std`
  2. Add `mean`
  3. Reverse channels if this option enabled.
  * `target_out` - super resolution model output layer name in case when model has several outputs.
* `multi_target_super_resolution` - converting output super resolution network with multiple outputs to `ContainerPrediction` with `SuperResolutionPrediction` for each output.
  * `reverse_channels` - allow switching output image channels e.g. RGB to BGR (Optional. Default value is False).
  * `mean` - value or list channel-wise values which should be added to result for getting values in range [0, 255] (Optional, default 0)
  * `std` - value or list channel-wise values on which result should be multiplied for getting values in range [0, 255] (Optional, default 255)
  * `cast_to_uint8` - perform casting output image pixels to [0, 255] range.
  **Important** Usually `mean` and `std` are the same which used in preprocessing, here they are used for reverting these preprocessing operations.
  The order of actions:
  1. Multiply on `std`
  2. Add `mean`
  3. Reverse channels if this option enabled.
  * `target_mapping` - dictionary where keys are a meaningful name for solved task which will be used as keys inside `ConverterPrediction`,  values - output layer names.
* `super_resolution_yuv` - converts output of super resolution model, which return output in YUV format, to `SuperResolutionPrediction`. Each output layer contains only 1 channel.
  * `y_output` - Y channel output layer.
  * `u_output` - U channel output layer.
  * `v_output` - V channel output layer.
  * `target_color` - taret color space for super resolution image - `bgr` and `rgb` are supported. (Optional, default `bgr`).
* `landmarks_regression` - converting output of model for landmarks regression to `FacialLandmarksPrediction`.
* `pixel_link_text_detection` - converting output of PixelLink like model for text detection to `TextDetectionPrediction`.
  * `pixel_class_out` - name of layer containing information related to text/no-text classification for each pixel.
  * `pixel_link_out` - name of layer containing information related to linkage between pixels and their neighbors.
  * `pixel_class_confidence_threshold` - confidence threshold for valid segmentation mask (Optional, default 0.8).
  * `pixel_link_confidence_threshold` - confidence threshold for valid pixel links (Optional, default 0.8).
  * `min_area` - minimal area for valid text prediction (Optional, default 0).
  * `min_height` - minimal height for valid text prediction (Optional, default 0).
* `ctpn_text_detection` - converting output of CTPN like model for text detection to `TextDetectionPrediction`.
  * `cls_prob_out` - name of output layer with class probabilities.
  * `bbox_pred_out` - name of output layer with predicted boxes.
  * `min_size` - minimal valid detected text proposals size (Optional, default 8).
  * `min_ratio` - minimal width / height ratio for valid text line (Optional, default 0.5).
  * `line_min_score` - minimal confidence for text line (Optional, default 0.9).
  * `text_proposals_width` - minimal width for text proposal (Optional, default 16).
  * `min_num_proposals` - minimal number for text proposals (Optional, default 2).
  * `pre_nms_top_n` - saved top n proposals before NMS applying (Optional, default 12000).
  * `post_nms_top_n` - saved top n proposals after NMS applying (Optional, default 1000).
  * `nms_threshold` - overlap threshold for NMS (Optional, default 0.7).
* `east_text_detection` - converting output of EAST like model for text detection to `TextDetectionPrediction`.
  * `score_map_out` - the name of output layer which contains score map.
  * `geometry_map_out` - the name of output layer which contains geometry map.
  * `score_map_threshold` - threshold for score map (Optional, default 0.8).
  * `nms_threshold` - threshold for text boxes NMS (Optional, default 0.2).
  * `box_threshold` - minimal confidence threshold for text boxes (Optional, default 0.1).
* `craft_text_detection` - converting output of CRAFT like model for text detection to `TextDetectionPrediction`.
  * `score_out` - the name of output layer which contains score map.
  * `text_threshold` - text confidence threshold (Optional, default 0.7).
  * `link_threshold` - link confidence threshold (Optional, default 0.4).
  * `low_text` - text low-bound score (Optional, default 0.4).
* `human_pose_estimation` - converting output of model for human pose estimation to `PoseEstimationPrediction`.
  * `part_affinity_fields_out` - name of output layer with keypoints pairwise relations (part affinity fields).
  * `keypoints_heatmap_out` - name of output layer with keypoints heatmaps.
  The output layers can be omitted if model has only one output layer - concatenation of this 2.
* `human_pose_estimation_openpose` - converting output of OpenPose-like model for human pose estimation to `PoseEstimationPrediction`.
  * `part_affinity_fields_out` - name of output layer with keypoints pairwise relations (part affinity fields).
  * `keypoints_heatmap_out` - name of output layer with keypoints heatmaps.
  * `upscale_factor` - upscaling factor for heatmaps and part affinity fields before post-processing.
* `human_pose_estimation_ae` - converting output of Associative Embedding-like model for human pose estimation to `PoseEstimationPrediction`.
  * `heatmaps_out` - name of output layer with keypoints heatmaps.
  * `nms_heatmaps_out` - name of output layer with keypoints heatmaps after non-maximum suppression.
  * `embeddings_out` - name of output layer with embedding (tag) maps.
* `beam_search_decoder` - realization CTC Beam Search decoder for symbol sequence recognition, converting model output to `CharacterRecognitionPrediction`.
  * `beam_size` -  size of the beam to use during decoding (default 10).
  * `blank_label` - index of the CTC blank label.
  * `softmaxed_probabilities` - indicator that model uses softmax for output layer (default False).
  * `logits_output` - Name of the output layer of the network to use in decoder
  * `custom_label_map` - Alphabet as a dict of strings. Must include blank symbol for CTC algorithm.
* `ctc_greedy_search_decoder` - realization CTC Greedy Search decoder for symbol sequence recognition, converting model output to `CharacterRecognitionPrediction`.
  * `blank_label` - index of the CTC blank label (default 0).
* `simple_decoder` - the easiest decoder for text recognition models, converts indices of classes to given letters, slices output on the first entry of `eos_label`
  * `eos_label` - label which should finish decoding
  * `custom_label_map` - label map (if not provided by the dataset meta)
* `ctc_beam_search_decoder` - Python implementation of CTC beam search decoder without LM for speech recognition.
* `ctc_greedy_decoder` - CTC greedy decoder for speech recognition.
* `ctc_beam_search_decoder_with_lm` - Python implementation of CTC beam search decoder with n-gram language model in kenlm binary format for speech recognition.
  * `beam_size` - Size of the beam to use during decoding (default 10).
  * `logarithmic_prob` - Set to "True" to indicate that network gives natural-logarithmic probabilities. Default is False for plain probabilities (after softmax).
  * `probability_out` - Name of the network's output with character probabilities (required)
  * `alphabet` - Alphabet as list of strings. Include an empty string for the CTC blank symbol. Default is space + 26 English letters + apostrophe + blank.
  * `sep` - Word separator character. Use an empty string for character-based LM. Default is space.
  * `lm_file` - Path to LM in binary kenlm format, relative to --model_attributes or --models.  Default is beam search without LM.
  * `lm_alpha` - LM alpha: weight factor for LM score (required when using LM)
  * `lm_beta` - LM beta: score bonus for each additional word, in log_e units (required when using LM)
  * `lm_oov_score` - Replace LM score for out-of-vocabulary words with this value (default -1000, ignored without LM)
  * `lm_vocabulary_offset` - Start of vocabulary strings section in the LM file.  Default is to not filter candidate words using vocabulary (ignored without LM)
  * `lm_vocabulary_length` - Size in bytes of vocabulary strings section in the LM file (ignored without LM)
* `fast_ctc_beam_search_decoder_with_lm` - CTC beam search decoder with n-gram language model in kenlm binary format for speech recognition, depends on `ctcdecode_numpy` Python module located in the `<omz_dir>/demos/speech_recognition_deepspeech_demo/python/ctcdecode-numpy/` directory.
  * `beam_size` - Size of the beam to use during decoding (default 10).
  * `logarithmic_prob` - Set to "True" to indicate that network gives natural-logarithmic probabilities. Default is False for plain probabilities (after softmax).
  * `probability_out` - Name of the network's output with character probabilities (required)
  * `alphabet` - Alphabet as list of strings. Include an empty string for the CTC blank sybmol. Default is space + 26 English letters + apostrophe + blank.
  * `sep` - Set to the empty string for character-based LM. Default is space.
  * `lm_file` - Path to LM in binary kenlm format, relative to --model_attributes or --models.  Default is beam search without LM.
  * `lm_alpha` - LM alpha: weight factor for LM score (required when using LM)
  * `lm_beta` - LM beta: score bonus for each additional word, in log_e units (required when using LM)
* `gaze_estimation` - converting output of gaze estimation model to `GazeVectorPrediction`.
* `hit_ratio_adapter` - converting output NCF model to `HitRatioPrediction`.
* `brain_tumor_segmentation` - converting output of brain tumor segmentation model to `BrainTumorSegmentationPrediction`.
  * `segmentation_out` - segmentation output layer name. (Optional, if not provided default first output blob will be used).
  * `make_argmax`  - allows applying argmax operation to output values. (default - `False`)
  * `label_order` - sets mapping from output classes to dataset classes. For example: `label_order: [3,1,2]` means that class with id 3 from model's output matches with class with id 1 from dataset,  class with id 1 from model's output matches with class with id 2 from dataset, class with id 2 from model's output matches with class with id 3 from dataset.
* `nmt` - converting output of neural machine translation model to `MachineTranslationPrediction`.
  * `vocabulary_file` - file which contains vocabulary for encoding model predicted indexes to words (e. g. vocab.bpe.32000.de). Path can be prefixed with `--models` arguments.
  * `eos_index` - index end of string symbol in vocabulary (Optional, used in cases when launcher does not support dynamic output shape for cut off empty prediction).
* `bert_question_answering_embedding` - converting output of BERT model trained to produce embedding vectors to `QuestionAnsweringEmbeddingPrediction`.
* `narnmt` - converting output of non-autoregressive neural machine translation model to `MachineTranslationPrediction`.
  * `vocabulary_file` - file which contains vocabulary for encoding model predicted indexes to words (e. g. vocab.json). Path can be prefixed with `--models` arguments.
  * `merges_file` - file which contains merges for encoding model predicted indexes to words (e. g. merges.txt). Path can be prefixed with `--models` arguments.
  * `output_name` - name of model's output layer if need (optional).
  * `sos_symbol` - string representation of start_of_sentence symbol (default=`<s>`).
  * `eos_symbol` - string representation of end_of_sentence symbol (default=`</s>`).
  * `pad_symbol` - string representation of pad symbol (default=`<pad>`).
  * `remove_extra_symbols` - remove sos/eos/pad symbols from predicted string (default=True)
* `bert_question_answering` - converting output of BERT model trained to solve question answering task to `QuestionAnsweringPrediction`.
* `bidaf_question_answering` - converting output of BiDAF model trained to solve question answering task to `QuestionAnsweringPrediction`.
  * `start_pos_output` - name of output layer with answer start position.
  * `end_pos_output` - name of output layer with answer end position.
* `bert_classification` - converting output of BERT model trained for text classification task to `ClassificationPrediction`.
  * `num_classes` - number of predicted classes.
  * `classification_out` - name of output layer with classification probabilities. (Optional, if not provided default first output blob will be used).
* `bert_ner` - converting output of BERT model trained for named entity recognition task to `SequenceClassificationPrediction`.
  * `classification_out` - name of output layer with classification probabilities. (Optional, if not provided default first output blob will be used).
* `human_pose_estimation_3d` - converting output of model for 3D human pose estimation to `PoseEstimation3dPrediction`.
  * `features_3d_out` - name of output layer with 3D coordinates maps.
  * `keypoints_heatmap_out` - name of output layer with keypoints heatmaps.
  * `part_affinity_fields_out` - name of output layer with keypoints pairwise relations (part affinity fields).
* `ctdet` - converting output of CenterNet object detection model to `DetectionPrediction`.
  * `center_heatmap_out` - name of output layer with center points heatmaps.
  * `width_height_out` - name of the output layer with object sizes.
  * `regression_out` - name of the regression output with the offset prediction.
* `mask_rcnn` - converting raw outputs of Mask-RCNN to combination of `DetectionPrediction` and `CoCoInstanceSegmentationPrediction`.
  * `classes_out` - name of output layer with information about classes (optional, if your model has detection_output layer as output).
  * `scores_out` - name of output layer with bbox scores (optional, if your model has detection_output layer as output).
  * `boxes_out` - name of output layer with bboxes (optional, if your model has detection_output layer as output).
  * `raw_masks_out` - name of output layer with raw instances masks.
  * `num_detections_out` - name of output layer with number valid detections (used in MaskRCNN models trained with TF Object Detection API).
  * `detection_out` - SSD-like detection output layer name (optional, if your model has scores_out, boxes_out and classes_out).
* `mask_rcnn_with_text` - converting raw outputs of Mask-RCNN with additional Text Recognition head to `TextDetectionPrediction`.
  * `classes_out` - name of output layer with information about classes.
  * `scores_out` - name of output layer with bbox scores.
  * `boxes_out` - name of output layer with bboxes.
  * `raw_masks_out` - name of output layer with raw instances masks.
  * `texts_out` - name of output layer with texts.
  * `confidence_threshold` - confidence threshold that is used to filter out detected instances.
* `yolact` - converting raw outputs of Yolact model to combination of `DetectionPrediction` and `CoCoInstanceSegmentationPrediction`.
  * `loc_out` - name of output layer which contains box locations, optional if boxes decoding embedded into model.
  * `prior_out` - name of output layer which contains prior boxes, optional if boxes decoding embedded into model.
  * `boxes_out` - name of output layer which contains decoded output boxes, optional if model has `prior` a `loc` outputs for boxes decoding.
  * `conf_out` - name of output layer which contains confidence scores for all classes for each box.
  * `mask_out` - name of output layer which contains instance masks.
  * `proto_out` - name of output layer which contains proto for masks calculation.
  * `confidence_threshold` - confidence threshold that is used to filter out detected instances (Optional, default 0.05).
  * `max_detections` - maximum detection used for metrics calculation (Optional, default 100).
* `class_agnostic_detection` - converting 'boxes' [n, 5] output of detection model to `DetectionPrediction` representation.
  * `output_blob` - name of output layer with bboxes.
  * `scale` - scalar value or list with 2 values to normalize bbox coordinates.
* `mono_depth` - converting output of monocular depth estimation model to `DepthEstimationPrediction`.
* `inpainting` - converting output of Image Inpainting model to `ImageInpaintingPrediction` representation.
* `style_transfer` - converting output of Style Transfer model to `StyleTransferPrediction` representation.
* `retinaface` - converting output of RetinaFace model to `DetectionPrediction` or representation container with `DetectionPrediction`, `AttributeDetectionPrediction`, `FacialLandmarksPrediction` (depends on provided set of outputs)
   * `scores_outputs` - the list of names for output layers with face detection score in order belonging to 32-, 16-, 8-strides.
   * `bboxes_outputs` - the list of names for output layers with face detection boxes in order belonging to 32-, 16-, 8-strides.
   * `landmarks_outputs` - the list of names for output layers with predicted facial landmarks in order belonging to 32-, 16-, 8-strides (optional, if not provided, only `DetectionPrediction` will be generated).
   * `type_scores_outputs` - the list of names for output layers with attributes detection score in order belonging to 32-, 16-, 8-strides (optional, if not provided, only `DetectionPrediction` will be generated).
   * `nms_threshold` - overlap threshold for NMS (optional, default 0.5).
   * `keep_top_k ` - maximal number of boxes which should be kept (optional).
   * `include_boundaries` - allows including boundaries for NMS (optional, default False).
* `retinaface-pytorch` - converting output of RetinaFace PyTorch model to `DetectionPrediction` or representation container with `DetectionPrediction`, `FacialLandmarksPrediction` (depends on provided set of outputs)
   * `scores_output` - name for output layer with face detection score.
   * `bboxes_output` - name for output layer with face detection boxes.
   * `landmarks_output` - name for output layer with predicted facial landmarks (optional, if not provided, only `DetectionPrediction` will be generated).
   * `nms_threshold` - overlap threshold for NMS (optional, default 0.4).
   * `keep_top_k ` - maximal number of boxes which should be kept (optional, default 750).
   * `include_boundaries` - allows including boundaries for NMS (optional, default False).
   * `confidence_threshold` - confidence threshold that is used to filter out detected instances (optional, default 0.02).
* `faceboxes` - converting output of FaceBoxes model to `DetectionPrediction` representation.
  * `scores_out` - name of output layer with bounding boxes scores.
  * `boxes_out` - name of output layer with bounding boxes coordinates.
* `prnet` - converting output of PRNet model for 3D landmarks regression task to `FacialLandmarks3DPrediction`
    * `landmarks_ids_file` - the file with indices for landmarks extraction from position heatmap. (Optional, default values defined [here](https://github.com/YadiraF/PRNet/blob/master/Data/uv-data/uv_kpt_ind.txt))
* `person_vehicle_detection` - converts output of person vehicle detection model to `DetectionPrediction` representation. Adapter merges scores, groups predictions into people and vehicles, and assigns labels accordingly.
    * `iou_threshold` - IOU threshold value for NMS operation.
* `face_detection` - converts output of face detection model to `DetectionPrediction ` representation. Operation is performed by mapping model output to the defined anchors, window scales, window translates, and window lengths to generate a list of face candidates.
    * `score_threshold` - Score threshold value used to discern whether a face is valid.
    * `layer_names` - Target output layer base names.
    * `anchor_sizes` - Anchor sizes for each base output layer.
    * `window_scales` - Window scales for each base output layer.
    * `window_lengths` - Window lengths for each base output layer.
* `face_detection_refinement` - converts output of face detection refinement model to `DetectionPrediction` representation. Adapter refines candidates generated in previous stage model.
    * `threshold` - Score threshold to determine as valid face candidate.
* `attribute_classification` - converts output of attributes classification model to `ContainerPrediction` which contains multiple `ClassificationPrediction` for attributes with their scores.
    * `output_layer_map` - dictionary where keys are output layer names of attribute classification model and values are the names of attributes.
* `regression` - converting output of regression model to `RegressionPrediction` representation.
    * `keep_shape` - allow keeping shape of predicted multi dimension array (Optional, default False).
* `multi_output_regression` - converting raw output features to `RegressionPrediction` for regression with gt data.
  * `output` - list of target output names.
* `mixed` - converts outputs of any model to `ContainerPrediction` which contains multiple types of predictions.
    * `adapters` - Dict where key is an output name and value is adapter config map including `output_blob` key to associate the output of model and this adapter.
* `person_vehilce_detection_refinement` - converts output of person vehicle detection refinement model to `DetectionPrediction` representation. Adapter refines proposals generated in previous stage model.
* `head_detection` - converts output of head detection model to `DetectionPrediction ` representation. Operation is performed by mapping model output to the defined anchors, window scales, window translates, and window lengths to generate a list of head candidates.
    * `score_threshold` - Score threshold value used to discern whether a face is valid.
    * `anchor_sizes` - Anchor sizes for each base output layer.
    * `window_scales` - Window scales for each base output layer.
    * `window_lengths` - Window lengths for each base output layer.
* `face_recognition_quality_assessment` - converts output of face recognition quality assessment model to `QualityAssessmentPrediction ` representation.
* `duc_segmentation` - converts output of DUC semantic segmentation model to `DUCSegmentationAdapter` representation
    * `ds_rate` - Specifies downsample rate.
    * `cell_width` - Specifies cell width to extract predictions.
    * `label_num` - Specifies number of output label classes.
* `stacked_hourglass` - converts output of Stacked Hourglass Networks for single human pose estimation to `PoseEstimationPrediction`.
   * `score_map_output`- the name of output layers for getting score map (Optional, default output blob will be used if not provided).
* `dna_seq_beam_search` - converts output of DNA sequencing model to `DNASequencePrediction` using beam search decoding.
  * `beam_size` - beam size for CTC Beam Search (Optional, default 5).
  * `threshold` - beam cut threshold (Optional, default 1e-3).
  * `output_blob` - name of output layer with sequence prediction.
* `pwcnet` - converts output of PWCNet network to `OpticalFlowPrediction`.
  * `flow_out` - target output layer name.
* `salient_object_detection` - converts output of salient object detection model to `SalientRegionPrediction`
  * `salient_map_output` - target output layer for getting salience map (Optional, if not provided default output blob will be used).
* `two_stage_detection` - converts output of 2-stage detector to `DetectionPrediction`.
  * `boxes_out` - output with bounding boxes in the format BxNx[x_min, y_min, width, height], where B - network batch size, N - number of detected boxes.
  * `cls_out` - output with classification probabilities in format [BxNxC], where B - network batch size, N - number of detected boxes, C - number of classed.
* `dumb_decoder` - converts  audio recognition model output to  `CharacterRecognitionPrediction`.
  * `alphabet` - model alphabet.
  * `uppercase` - produce prediction in uppercase, default is `True`.
* `detr` - converts output of DETR models family to `DetectionPrediction`.
    * `scores_out` - output layer name with detection scores logits.
    * `boxes_out` - output layer name with detection boxes coordinates in [Cx,Cy,W, H] format, where Cx - x coordinate of box center, Cy - y coordinate of box center, W, H - width and height respectively.
* `ultra_lightweight_face_detection` - converts output of Ultra-Lightweight Face Detection models to `DetectionPrediction` representation.
  * `scores_out` - name of output layer with bounding boxes scores.
  * `boxes_out` - name of output layer with bounding boxes coordinates.
  * `score_threshold` - minimal accepted score for valid boxes (Optional, default 0.7).
* `trimap` - converts greyscale model output to `ImageProcessingPrediction`. Replaces pixel values in cut and keep zones with 0 and 1 respectively. All other postprocessing inherited from `image_processing` adapter.
* `background_matting` - converts output of background matting model to `BackgroundMattingPrediction`.
* `noise_suppression` - converts output of audio denoising model to `NoiseSuppressionPrediction`.
  * `output_blob` - name of output layer with processed signal (Optional, if not provided, first found output from model will be used).
* `kaldi_latgen_faster_mapped` - decodes output Kaldi\* automatic speech recognition model using lattice generation approach with transition model to `CharcterRecognitionPrediction`.
  **Important note** This adapter requires [Kaldi\* installation](https://kaldi-asr.org/doc/install.html)(we recommend to use `67db30cc` commit)
  and providing path to directory with compiled executable apps: `latgen-faster-mapped`, `lattice-scale`, `lattice-add-penalty`, `lattice-best-path`.
  Path directory can be provided using `--kaldi_bin_dir` commandline argument or `KALDI_BIN_DIR` environment variable.
  * `fst_file` - Weighted Finite-State Transducers (WFST) state graph file.
  *`words_file` - words table file.
  * `transition_model_file` - transition model file.
  * `beam` - beam size (Optional, default `1`).
  * `lattice_beam` - lattice beam size (Optional, default `1`).
  * `allow_partial` - allow partial decoding (Optional, default `False`).
  * `acoustic_scale` - acoustic scale for decoding (Optional, default `0.1`).
  * `min_active` - min active paths for decoding (Optional, default `200`).
  * `max_active` - max active paths for decoding (Optional, default `7000`).
  * `inverse_acoustic_scale` - inverse acoustic scale for lattice scaling (Optional, default `0`).
  * `word_insertion_penalty` - add word insertion penalty to the lattice. Penalties are negative log-probs, base e, and are added to the language model' part of the cost (Optional, `0`).
* `quantiles_predictor` - converts output of Time Series Forecasting models to `TimeSeriesForecastingQuantilesPrediction`.
  * `quantiles` - predictions[i]->quantile[i] mapping.
  * `output_name` - name of output node to convert.
