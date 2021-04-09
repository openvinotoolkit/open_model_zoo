# netvlad-tf

## Use Case and High-Level Description

NetVLAD is a CNN architecture which tackles the problem of large scale visual place recognition. The architecture uses VGG 16 as base network and NetVLAD - a new trainable generalized VLAD (Vector of Locally Aggregated Descriptors) layer. It is a place recognition model pre-trained on the [Pittsburgh 250k](http://www.ok.ctrl.titech.ac.jp/~torii/project/repttile/) dataset.

For details see [repository](https://github.com/uzh-rpg/netvlad_tf_open) and [paper](https://arxiv.org/abs/1511.07247).

## Specification

| Metric            | Value             |
|-------------------|-------------------|
| Type              | Place recognition |
| GFLOPs            | 36.6374           |
| MParams           | 149.0021          |
| Source framework  | TensorFlow\*      |

## Accuracy

Accuracy metrics are obtained on a smaller validation subset of Pittsburgh 250k dataset (Pitts30k) containing 10k database images in each set (train/test/validation).  Images were resized to input size.

| Metric              | Value   |
| ------------------- | ------- |
| localization_recall | 82.0321%|

## Input

### Original model

Image, name - `Placeholder`,  shape - `1, 200, 300, 3`, format is `B, H, W, C`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `RGB`.

### Converted model

Image, name - `Placeholder`,  shape - `1, 3, 200, 300`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original model

Floating point embeddings, name - `vgg16_netvlad_pca/l2_normalize_1`,  shape - `1, 4096`, output data format  - `B, C`, where:

- `B` - batch size
- `C` - vector of 4096 floating points values, local image descriptors

### Converted model

Floating point embeddings, name - `vgg16_netvlad_pca/l2_normalize_1`,  shape - `1, 4096`, output data format  - `B, C`, where:

- `B` - batch size
- `C` - vector of 4096 floating points values, local image descriptors

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

The original model is distributed under
[MIT license](https://raw.githubusercontent.com/uzh-rpg/netvlad_tf_open/master/LICENSE):

```
MIT License

Copyright (c) 2018 Robotics and Perception Group

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
