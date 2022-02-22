# common-sign-language-0001

## Use Case and High-Level Description

A human gesture recognition model for the Jester dataset recognition scenario
(gesture-level recognition). The model uses an S3D framework with MobileNet V3 backbone.
Please refer to the [Jester](https://web.archive.org/web/20210722062232/https://20bn.com/datasets/jester/) dataset specification
to see the list of gestures that are recognized by this model.

The model accepts a stack of frames (8 frames) sampled with a constant frame rate (15 FPS)
and produces a prediction on the input clip.

## Specification

| Metric                                  | Value        |
|-----------------------------------------|--------------|
| Top-1 accuracy (continuous Jester)      | 93.58%       |
| GFlops                                  | 4.2269       |
| MParams                                 | 4.1128       |
| Source framework                        | PyTorch\*    |

## Input

### Original Model

Batch of images of the shape `1, 3, 8, 224, 224` in the `B, C, T, H, W` format, where:

- `B` - batch size
- `C` - channel
- `T` - sequence length
- `H` - height
- `W` - width

Channel order is `RGB`.

### Converted Model

Batch of images of the shape `1, 3, 8, 224, 224` in the `B, C, T, H, W` format, where:

- `B` - batch size
- `C` - channel
- `T` - sequence length
- `H` - height
- `W` - width

Channel order is `RGB`.

## Output

The model outputs a tensor with the shape `B, 27`, each row is a logits vector of performed Jester gestures.

### Original Model

Blob of the shape `1, 27` in the `B, C` format, where:

- `B` - batch size
- `C` - predicted logits size

### Converted Model

Blob of the shape `1, 27` in the `B, C` format, where:

- `B` - batch size
- `C` - predicted logits size

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

* [G-API Gesture Recognition Demo](../../../demos/gesture_recognition_demo/cpp_gapi/README.md)
* [Gesture Recognition Python\* Demo](../../../demos/gesture_recognition_demo/python/README.md)

## Legal Information
[\*] Other names and brands may be claimed as the property of others.

The original model is distributed under the
[Apache License 2.0](https://github.com/sovrasov/mmaction2/blob/ote/LICENSE).
