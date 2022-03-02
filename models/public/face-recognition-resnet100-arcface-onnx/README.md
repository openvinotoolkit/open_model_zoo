# face-recognition-resnet100-arcface-onnx

## Use Case and High-Level Description

The `face-recognition-resnet100-arcface-onnx` model is a deep face recognition model with ResNet100 backbone and ArcFace loss. ArcFace is a novel supervisor signal called additive angular margin which used as an additive term in the softmax loss to enhance the discriminative power of softmax loss.
This model is pre-trained in MXNet\* framework and converted to ONNX\* format. More details provided in the [paper](https://arxiv.org/abs/1801.07698) and [repository](https://github.com/onnx/models/tree/master/vision/body_analysis/arcface).

## Specification

| Metric            | Value            |
|-------------------|------------------|
| Type              | Face recognition |
| GFLOPs            | 24.2115          |
| MParams           | 65.1320          |
| Source framework  | MXNet\*          |

## Accuracy

| Metric      | Value |
| ----------- | ----- |
| LFW accuracy| 99.68%|

## Input

### Original Model

Image, name: `data`,  shape: `1, 3, 112, 112`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `RGB`.

### Converted Model

Image, name: `data`,  shape: `1, 3, 112, 112`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original Model

Face embeddings, name: `fc1`,  shape: `1, 512`, output data format: `B, C`, where:

- `B` - batch size
- `C` - row-vector of 512 floating points values, face embeddings

The net outputs on different images are comparable in cosine distance.

### Converted Model

Face embeddings, name: `fc1`,  shape: `1, 512`, output data format: `B, C`, where:

- `B` - batch size
- `C` - row-vector of 512 floating points values, face embeddings

The net outputs on different images are comparable in cosine distance.

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

* [Face Recognition Python\* Demo](../../../demos/face_recognition_demo/python/README.md)
* [Smart Classroom C++ Demo](../../../demos/smart_classroom_demo/cpp/README.md)
* [Smart Classroom C++ G-API Demo](../../../demos/smart_classroom_demo/cpp_gapi/README.md)

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/onnx/models/master/LICENSE).
A copy of the license is provided in `<omz_dir>/models/public/licenses/APACHE-2.0.txt`.
