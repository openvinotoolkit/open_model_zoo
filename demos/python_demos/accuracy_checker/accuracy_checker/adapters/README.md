# Adapters

Adapter is a function for conversion network infer output to metric specific format.
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

AccuracyChecker supports following set of adapters:
* `classification` - converting output of classification model to `ClassificationPrediction` representation.
* `segmentation` - converting output of semantic segmentation model to `SeegmentationPrediction` representation.
* `tiny_yolo_v1` - converting output of Tiny YOLO v1 model to `DetectionPrediction` representation.
* `reid` - converting output of reidentification model to `ReIdentificationPrediction` representation.
  * `grn_workaround` - enabling processing output with adding Global Region Normalization layer.
* `yolo_v2` - converting output of YOLO v2 model to `DetectionPrediction` representation.
* `lpr` - converting output of license plate recognition model to `CharacterRecognitionPrediction` representation.
* `ssd - converting  output of SSD model to `DetectionPrediction` representation.
* `face_person_detection` - converting face person detection model output with 2 detection outputs to `ContainerPredition`, where value of parameters `face_out`and `person_out` are used for identification `DetectionPrediction` in container. 
  * `face_out` -  face detection output layer name.
  * `person_out` - person detection output layer name.
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
* `landmarks_regression` - converting output of model for landmarks regression to `PointRegressionPrediction`.
