# facenet-20180408-102900

## Use Case and High-Level Description

FaceNet: A Unified Embedding for Face Recognition and Clustering. For details see the [repository](https://github.com/davidsandberg/facenet/), [paper](https://arxiv.org/abs/1503.03832)

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Face recognition                          |
| GFlops                          | 2.846                                     |
| MParams                         | 23.469                                    |
| Source framework                | TensorFlow\*                              |

## Accuracy

| Metric      | Value |
| ----------- | ----- |
| LFW accuracy| 99.14%|

## Input

### Original model

1. Image, name - `batch_join:0`, shape - `1, 160, 160, 3`, format `B, H, W, C`, where:

    - `B` - batch size
    - `H` - image height
    - `W` - image width
    - `C` - number of channels

   Expected color order - `RGB`.
   Mean values - [127.5, 127.5, 127.5], scale factor for each channel - 128.0

2. A boolean input, manages state of the graph (train/infer), name - `phase_train`, shape - `1`.

### Converted model

Image, name - `image_batch/placeholder_port_0`, shape - `1, 3, 160, 160`, format `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: `BGR`.

## Output

### Original model

Vector of floating-point values - face embeddings, Name - `embeddings`.

### Converted model

Face embeddings, name - `InceptionResnetV1/Bottleneck/BatchNorm/Reshape_1/Normalize`, in format `B,C`, where:

- `B` - batch size
- `C` - row-vector of 512 floating-point values - face embeddings

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
[license](https://raw.githubusercontent.com/davidsandberg/facenet/master/LICENSE.md):

```
MIT License

Copyright (c) 2016 David Sandberg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
