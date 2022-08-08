# ERFNet

## Use Case and High-Level Description

  This is a ONNX* version of `erfnet` model designed to perform real-time lane detection.
  This model is pre-trained in PyTorch* framework and retrained by CULane.
  For details see [repository](https://github.com/Zhangxianwen2021/ERFNet),
  paper of [ERFNet](https://doi.org/10.1109/TITS.2017.2750080)and [repository](https://github.com/cardwing/Codes-for-Lane-Detection/tree/master/ERFNet-CULane-PyTorch)

## Specification

| Metric            |        Value          |
|-------------------|-----------------------|
| Type              | Semantic segmentation |
| GFLOPs            | 11.13                 |
| MParams           | 7.87                  |
| Source framework  | PyTorch*              |

## Accuracy

|  Metric  |  Value |
|  ------  | -------|
| accuracy | 97.71% |

## Input

### Original model

Image, name - `input_1`, shape - `1,3,208,976`, format is `B,C,H,W` where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `RGB`.
Scale value - 255.

### Converted model

Image, name - `input_1`, shape - `1,3,208,976`, format is `B,C,H,W` where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `BGR`.

## Output

### Original model
Image, name - `output1`, shape - `1,5,208,976`, format is `B,C,H,W` where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `BGR`.

### Converted model

Image, name - `output1`, shape - `1,5,208,976`, format is `B,C,H,W` where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `BGR`.

## Download a Model and Convert it into Inference Engine Format

You can download models and if necessary convert them into Inference Engine format using the Model Downloader and other automation tools as shown in the examples below.
An example of using the Model Downloader:

python3 <omz_dir>/tools/downloader/downloader.py --name <model_name>

An example of using the Model Converter:

python3 <omz_dir>/tools/downloader/converter.py --name <model_name>

## Demo usage

The model can be used in the following demos provided by the Open Model Zoo to show its capabilities:

* [Lane Detection Python\* Demo](../../../demos/lane_detection_demo/python/README.md)
## Legal Information

The original model is distributed under the following [license1](https://raw.githubusercontent.com/onnx/models/master/LICENSE).
A copy of the license is provided in <omz_dir>/models/public/licenses/APACHE-2.0.txt.


