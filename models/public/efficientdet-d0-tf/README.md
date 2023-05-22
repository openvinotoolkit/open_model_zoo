# efficientdet-d0-tf

## Use Case and High-Level Description

The `efficientdet-d0-tf` model is one of the [EfficientDet](https://arxiv.org/abs/1911.09070)
models  designed to perform object detection. This model was pre-trained in TensorFlow\*.
All the EfficientDet models have been pre-trained on the [Common Objects in Context (COCO)](https://cocodataset.org/#home) image database.
For details about this family of models, check out the Google AutoML [repository](https://github.com/google/automl/tree/master/efficientdet).

## Steps to Reproduce Conversion to Frozen Graph

1. Clone the original repository
```sh
git clone https://github.com/google/automl.git
cd automl
```
2. Checkout the commit that the conversion was tested on:
```sh
git checkout 341af7d4da7805c3a874877484e133f33c420ec5
```
3. Navigate to efficientdet source code directory
```sh
cd efficientdet
```
4. Install dependencies
```sh
pip install -r requirements.txt
```
5. Download model checkpoint archive using this [link](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d0.tar.gz) and unzip it.
6. Run following command:
   ```sh
   python model_inspect.py --runmode=saved_model --model_name=efficientdet-d0 --ckpt_path=CHECKPOINT_DIR --saved_model_dir=OUTPUT_DIR
   ```
   where `CHECKPOINT_DIR` - directory where model checkpoint stored, `OUTPUT_DIR` - directory where converted model should be stored.

## Specification

| Metric            | Value           |
|-------------------|-----------------|
| Type              | Object detection|
| GFLOPs            |     2.54        |
| MParams           |     3.9         |
| Source framework  | TensorFlow\*    |

## Accuracy

| Metric                                                                | Converted model |
| --------------------------------------------------------------------- | --------------- |
| [COCO mAP (0.5:0.05:0.95)](https://cocodataset.org/#detection-eval)   | 31.95%          |

## Input

### Original Model

Image, name - `image_arrays`,  shape - `1, 512, 512, 3`, format is `B, H, W, C`, where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `RGB`.

### Converted Model

Image, name - `image_arrays/placeholder_port_0`,  shape - `1, 512, 512, 3`, format is `B, H, W, C`, where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `BGR`.

## Output

### Original Model

The array of summary detection information, name: `detections`, shape: `1, 100, 7` in the format  `1, N, 7`, where `N` is the number of detected
bounding boxes. For each detection, the description has the format:
[`image_id`, `y_min`, `x_min`, `y_max`, `x_max`, `confidence`, `label`], where:

- `image_id` - ID of the image in the batch
- (`x_min`, `y_min`) - coordinates of the top left bounding box corner
- (`x_max`, `y_max`) - coordinates of the bottom right bounding box corner
- `confidence` - confidence for the predicted class
- `label` - predicted class ID, in range [1, 91], mapping to class names provided in `<omz_dir>/data/dataset_classes/coco_91cl.txt` file

### Converted Model

The array of summary detection information, name: `detections`, shape: `1, 1, 100, 7` in the format `1, 1, N, 7`, where `N` is the number of detected
bounding boxes. For each detection, the description has the format:
[`image_id`, `label`, `conf`, `x_min`, `y_min`, `x_max`, `y_max`], where:

- `image_id` - ID of the image in the batch
- `label` - predicted class ID, in range [0, 90], mapping to class names provided in `<omz_dir>/data/dataset_classes/coco_91cl.txt` file
- `conf` - confidence for the predicted class
- (`x_min`, `y_min`) - coordinates of the top left bounding box corner (coordinates stored in normalized format, in range [0, 1])
- (`x_max`, `y_max`) - coordinates of the bottom right bounding box corner  (coordinates stored in normalized format, in range [0, 1])

## Download a Model and Convert it into OpenVINO™ IR Format

You can download models and if necessary convert them into OpenVINO™ IR format using the [Model Downloader and other automation tools](../../../tools/model_tools/README.md) as shown in the examples below.

An example of using the Model Downloader:
```
omz_downloader --name <model_name>
```

An example of using the Model Converter:
```
omz_converter --name <model_name>
```

## Demo usage

The model can be used in the following demos provided by the Open Model Zoo to show its capabilities:

* [Object Detection C++ Demo](../../../demos/object_detection_demo/cpp/README.md)
* [Object Detection Python\* Demo](../../../demos/object_detection_demo/python/README.md)
* [Pedestrian Tracker C++ Demo](../../../demos/pedestrian_tracker_demo/cpp/README.md)

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/google/automl/master/LICENSE).
A copy of the license is provided in `<omz_dir>/models/public/licenses/APACHE-2.0-TF-AutoML.txt`.
