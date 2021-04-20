# mask_rcnn_resnet50_atrous_coco

## Use Case and High-Level Description

Mask R-CNN ResNet50 Atrous trained on [Common Objects in Context (COCO)](https://cocodataset.org/#home) dataset. It is used for object instance segmentation.
For details, see the [paper](https://arxiv.org/abs/1703.06870).

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Instance segmentation                     |
| GFlops                          | 294.738                                   |
| MParams                         | 50.222                                    |
| Source framework                | TensorFlow\*                              |

## Accuracy

| Metric                   | Value  |
| ------------------------ | ------ |
| coco_orig_precision      | 29.75% |
| coco_orig_segm_precision | 27.46% |

## Input

### Original Model

Image, name: `image_tensor`, shape: `1, 800, 1365, 3`, format: `B, H, W, C`, where:

- `B` - batch size
- `H` - image height
- `W` - image width
- `C` - number of channels

Expected color order: `RGB`.

### Converted Model

1. Image, name: `image_tensor`, shape: `1, 3, 800, 1365`, format: `B, C, H, W`, where:

    - `B` - batch size
    - `C` - number of channels
    - `H` - image height
    - `W` - image width

    Expected color order: `BGR`.

2. Information of input image size, name: `image_info`, shape: `1, 3`, format: `B, C`, where:

    - `B` - batch size
    - `C` - vector of 3 values in format `H, W, S`, where `H` is an image height, `W` is an image width, `S` is an image scale factor (usually 1)

## Output

### Original Model

1. Classifier, name: `detection_classes`. Contains predicted bounding-boxes classes in a range [1, 91]. The model was trained on [Common Objects in Context (COCO)](https://cocodataset.org/#home) dataset version with 90 categories of objects, 0 class is for background.
2. Probability, name: `detection_scores`. Contains probability of detected bounding boxes.
3. Detection box, name: `detection_boxes`. Contains detection boxes coordinates in a format `[y_min, x_min, y_max, x_max]`, where (`x_min`, `y_min`)  are coordinates of the top left corner, (`x_max`, `y_max`) are coordinates of the right bottom corner. Coordinates are rescaled to input image size.
4. Detections number, name: `num_detections`. Contains the number of predicted detection boxes.
5. Segmentation mask, name: `detection_masks`. Contains segmentation heatmaps of detected objects for all classes for every output bounding box.

### Converted Model

1. The array of summary detection information, name: `reshape_do_2d`, shape: `100, 7` in the format `N, 7`, where `N` is the number of detected
bounding boxes. For each detection, the description has the format:
[`image_id`, `label`, `conf`, `x_min`, `y_min`, `x_max`, `y_max`], where:

    - `image_id` - ID of the image in the batch
    - `label` - predicted class ID
    - `conf` - confidence for the predicted class
    - (`x_min`, `y_min`) - coordinates of the top left bounding box corner (coordinates stored in normalized format, in range [0, 1])
    - (`x_max`, `y_max`) - coordinates of the bottom right bounding box corner  (coordinates stored in normalized format, in range [0, 1])

2. Segmentation heatmaps for all classes for every output bounding box, name: `masks`, shape: `100, 90, 33, 33` in the format `N, 90, 33, 33`, where `N` is the number of detected masks, 90 is the number of classes (the background class excluded).

## Download a Model and Convert it into Inference Engine Format

You can download models and if necessary convert them into Inference Engine format using the [Model Downloader and other automation tools](../../../tools/downloader/README.md) as shown in the examples below.

An example of using the Model Downloader:
```
python3 <omz_dir>/tools/downloader/downloader.py --name <model_name>
```

An example of using the Model Converter:
```
python3 <omz_dir>/tools/downloader/converter.py --name <model_name>
```

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/tensorflow/models/master/LICENSE).
A copy of the license is provided in [APACHE-2.0-TF-Models.txt](../licenses/APACHE-2.0-TF-Models.txt).
