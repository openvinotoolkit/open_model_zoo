# retinanet-tf

## Use Case and High-Level Description

RetinaNet is the dense object detection model with ResNet50 backbone, originally trained on Keras\*, then
converted to TensorFlow\* protobuf format. For details, see [paper](https://arxiv.org/abs/1708.02002),
[repository](https://github.com/fizyr/keras-retinanet).

### Steps to Reproduce Keras\* to TensorFlow\* Conversion

1. Clone and install the original [repository](https://github.com/fizyr/keras-retinanet) (tested on `47fdf189` commit) following the [installation steps](https://github.com/fizyr/keras-retinanet#installation). Make sure you ran `python setup.py build_ext --inplace` from the root of repository.
2. Download the original model from [here](https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5)
3. Get conversion script:
   1. Get conversion script from [repository](https://github.com/amir-abdi/keras_to_tensorflow):
   ```sh
   git clone https://github.com/amir-abdi/keras_to_tensorflow.git
   ```
   2. Checkout the commit that the conversion was tested on:
   ```
   git checkout c841508a88faa5aa1ffc7a4947c3809ea4ec1228
   ```
   3. Apply `keras_to_tensorflow.patch` from `<omz_dir>/models/public/retinanet-tf`:
   ```
   git apply keras_to_tensorflow.patch
   ```
   4. Run script:
   ```
   python keras_to_tensorflow.py --input_model=<model_in>.h5 --output_model=<model_out>.pb
   ```
> **NOTE**: The conversion was tested with keras==2.3.1, tensorflow==1.15 and h5py<3.0.0.

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Object detection                          |
| GFlops                          | 238.9469                                  |
| MParams                         | 64.9706                                   |
| Source framework                | TensorFlow\*                              |

## Accuracy

| Metric         | Value |
| -------------- | ----- |
| coco_precision | 33.15%|

## Input

### Original Model

Image, name: `input_1`, shape: `1, 1333, 1333, 3`, format: `B, H, W, C`, where:

- `B` - batch size
- `H` - image height
- `W` - image width
- `C` - number of channels

Expected color order: `BGR`.
Mean values: [103.939, 116.779, 123.68]

### Converted Model

Image, name: `input_1`, shape: `1, 1333, 1333, 3`, format: `B, H, W, C`, where:

- `B` - batch size
- `H` - image height
- `W` - image width
- `C` - number of channels

Expected color order: `BGR`.

## Output

### Original Model

1. Classifier, name: `filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3`. Contains predicted bounding boxes classes in a range [1, 80]. The model was trained on [Common Objects in Context (COCO)](https://cocodataset.org/#home) dataset version with 80 categories of objects, 0 class is for background. Mapping to class names provided in `<omz_dir>/data/dataset_classes/coco_80cl_bkgr.txt` file
2. Probability, name: `filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3`. Contains probability of detected bounding boxes.
3. Detection box, name: `filtered_detections/map/TensorArrayStack/TensorArrayGatherV3`. Contains detection boxes coordinates in a format `[y_min, x_min, y_max, x_max]`, where (`x_min`, `y_min`)  are coordinates of the top left corner, (`x_max`, `y_max`) are coordinates of the right bottom corner. Coordinates are rescaled to input image size.

### Converted Model

The array of summary detection information, name - `DetectionOutput`, shape - `1, 1, 300, 7` in the format `1, 1, N, 7`, where `N` is the number of detected
bounding boxes. For each detection, the description has the format:
[`image_id`, `label`, `conf`, `x_min`, `y_min`, `x_max`, `y_max`], where:

- `image_id` - ID of the image in the batch
- `label` - predicted class ID in range [1, 80], mapping to class names provided in `<omz_dir>/data/dataset_classes/coco_80cl_bkgr.txt` file
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
[Apache License, Version 2.0](https://raw.githubusercontent.com/fizyr/keras-retinanet/master/LICENSE).
A copy of the license is provided in `<omz_dir>/models/public/licenses/APACHE-2.0.txt`.
