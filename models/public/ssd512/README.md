# ssd512

## Use Case and High-Level Description

The `ssd512` model is the Caffe\* framework implementation of [Single-Shot multibox Detection (SSD)](https://arxiv.org/abs/1512.02325) algorithm with 512x512 input resolution and `VGG-16` backbone. The network intended to perform visual object detection. This model is pretrained on VOC2007 + VOC2012 + COCO dataset and is able to detect 20 [PASCAL VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) object classes:

- Person: person
- Animal: bird, cat, cow, dog, horse, sheep
- Vehicle: aeroplane, bicycle, boat, bus, car, motorbike, train
- Indoor: bottle, chair, dining table, potted plant, sofa, tv/monitor

Mapping model labels to class names provided in `<omz_dir>/data/dataset_classes/voc_20cl_bkgr.txt` file.

For details about this model, check out the [repository](https://github.com/weiliu89/caffe/tree/ssd).

## Example

See [here](https://github.com/weiliu89/caffe/tree/ssd).

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Detection     |
| GFLOPs            | 180.611       |
| MParams           | 27.189        |
| Source framework  | Caffe\*       |

## Accuracy

The accuracy results were obtained on test data from VOC2007 dataset.

| Metric | Value  |
| ------ | ------ |
| mAP    | 91.07% |

## Input

### Original model

Image, name - `data`, shape - `1, 3, 512, 512`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.
Mean values - [104.0, 117.0, 123.0]

### Converted model

Image, name - `data`, shape - `1, 3, 512, 512`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original model

The array of detection summary info, name - `detection_out`, shape - `1, 1, 200, 7` in the format `1, 1, N, 7`, where `N` is the number of detected bounding boxes. For each detection, the description has the format:
[`image_id`, `label`, `conf`, `x_min`, `y_min`, `x_max`, `y_max`], where:

- `image_id` - ID of the image in the batch
- `label` - predicted class ID (1..20 - PASCAL VOC defined class ids). Mapping to class names provided by `<omz_dir>/data/dataset_classes/voc_20cl_bkgr.txt` file.
- `conf` - confidence for the predicted class, in [0, 1] range
- (`x_min`, `y_min`) - coordinates of the top left bounding box corner (coordinates are in normalized format, in range [0, 1])
- (`x_max`, `y_max`) - coordinates of the bottom right bounding box corner (coordinates are in normalized format, in range [0, 1])

### Converted model

The array of detection summary info, name - `detection_out`, shape - `1, 1, 200, 7` in the format `1, 1, N, 7`, where `N` is the number of detected bounding boxes. For each detection, the description has the format:
[`image_id`, `label`, `conf`, `x_min`, `y_min`, `x_max`, `y_max`], where:

- `image_id` - ID of the image in the batch
- `label` - predicted class ID (1..20 - PASCAL VOC defined class ids). Mapping to class names provided by `<omz_dir>/data/dataset_classes/voc_20cl_bkgr.txt` file.
- `conf` - confidence for the predicted class, in [0, 1] range
- (`x_min`, `y_min`) - coordinates of the top left bounding box corner (coordinates are in normalized format, in range [0, 1])
- (`x_max`, `y_max`) - coordinates of the bottom right bounding box corner (coordinates are in normalized format, in range [0, 1])

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

The original model is distributed under the following
[license](https://raw.githubusercontent.com/weiliu89/caffe/ssd/LICENSE):

```
COPYRIGHT

All new contributions compared to the original branch:
Copyright (c) 2015, 2016 Wei Liu (UNC Chapel Hill), Dragomir Anguelov (Zoox),
Dumitru Erhan (Google), Christian Szegedy (Google), Scott Reed (UMich Ann Arbor),
Cheng-Yang Fu (UNC Chapel Hill), Alexander C. Berg (UNC Chapel Hill).
All rights reserved.

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.

Caffe uses a shared copyright model: each contributor holds copyright over
their contributions to Caffe. The project versioning records all such
contribution and copyright details. If a contributor wants to further mark
their specific copyright on a particular contribution, they should indicate
their copyright solely in the commit message of the change when it is
committed.

LICENSE

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

CONTRIBUTION AGREEMENT

By contributing to the BVLC/caffe repository through pull-request, comment,
or otherwise, the contributor releases their content to the
license and copyright terms herein.
```
