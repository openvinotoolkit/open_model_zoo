# retinanet-tf

## Use Case and High-Level Description

RetinaNet is the dense object detection model with ResNet50 backbone, originally trained on Keras\*, then
converted to TensorFlow\* protobuf format. For details, see [paper](https://arxiv.org/abs/1708.02002),
[repository](https://github.com/fizyr/keras-retinanet).

### Steps to Reproduce Keras\* to TensorFlow\* Conversion

1. Clone the original [repository](https://github.com/fizyr/keras-retinanet)(tested on `47fdf189` commit)
2. Download the original model from [here](https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5)
4. Get conversion script:

    1. Get conversion script from [repository](https://github.com/amir-abdi/keras_to_tensorflow):
        ```sh
        git clone https://github.com/amir-abdi/keras_to_tensorflow.git
        ```
    1. (Optional) Checkout the commit that the conversion was tested on:
        ```
        git checkout c841508a88faa5aa1ffc7a4947c3809ea4ec1228
        ```
    1. Apply `keras_to_tensorflow.patch`:
        ```
        git apply keras_to_tensorflow.patch
        ```
    1. Run script:
        ```
        python keras_to_tensorflow.py --input_model=<model_in>.h5 --output_model=<model_out>.pb
        ```

## Example

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Object detection                          |
| GFlops                          | 238.9469                                  |
| MParams                         | 64.9706                                   |
| Source framework                | TensorFlow\*                              |

## Accuracy

| Metric | Value |
| ------ | ----- |
| coco_precision | 33.15%|

## Performance

## Input

### Original Model

Image, name: `input_1`, shape: [1x1333x1333x3], format: [BxHxWxC], where:

- B - batch size
- H - image height
- W - image width
- C - number of channels

Expected color order: BGR.
Mean values: [103.939, 116.779, 123.68]

### Converted Model

Image, name: `input_1`, shape: [1x3x1333x1333], format: [BxCxHxW], where:

- B - batch size
- C - number of channels
- H - image height
- W - image width

Expected color order: BGR.

## Output

### Original Model

1. Classifier, name: `filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3`. Contains predicted bounding boxes classes in a range [1, 80]. The model was trained on the Microsoft\* COCO dataset version with 80 categories of objects.
2. Probability, name: `filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3`. Contains probability of detected bounding boxes.
3. Detection box, name: `filtered_detections/map/TensorArrayStack/TensorArrayGatherV3`. Contains detection boxes coordinates in a format `[y_min, x_min, y_max, x_max]`, where (`x_min`, `y_min`)  are coordinates of the top left corner, (`x_max`, `y_max`) are coordinates of the right bottom corner. Coordinates are rescaled to input image size.

### Converted Model

The array of summary detection information, name - `DetectionOutput`, shape - [1, 1, N, 7], where N is the number of detected
bounding boxes. For each detection, the description has the format:
[`image_id`, `label`, `conf`, `x_min`, `y_min`, `x_max`, `y_max`],
    where:

    - `image_id` - ID of the image in the batch
    - `label` - predicted class ID
    - `conf` - confidence for the predicted class
    - (`x_min`, `y_min`) - coordinates of the top left bounding box corner (coordinates stored in normalized format, in range [0, 1])
    - (`x_max`, `y_max`) - coordinates of the bottom right bounding box corner  (coordinates stored in normalized format, in range [0, 1])

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/fizyr/keras-retinanet/master/LICENSE).
A copy of the license is provided in [APACHE-2.0.txt](../licenses/APACHE-2.0.txt).
