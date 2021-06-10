# f3net

## Use Case and High-Level Description

F3Net: Fusion, Feedback and Focus for Salient Object Detection. For details see
the [repository](https://github.com/weijun88/F3Net), [paper](https://arxiv.org/abs/1911.11445)

## Specification

| Metric                          | Value                                    |
|---------------------------------|------------------------------------------|
| Type                            | Salient object detection                 |
| GFLOPs                          | 31.2883                                  |
| MParams                         | 25.2791                                  |
| Source framework                | PyTorch\*                                |

## Accuracy

| Metric    | Value |
| --------- | ----- |
| F-measure | 84.21%|

The F-measure estimated on [Pascal-S](http://cbs.ic.gatech.edu/salobj/) dataset and defined as the weighted harmonic mean of precision and recall.

`F-measure` = `(1 + β^2) * (Precision * Recall) / (β^2 * (Precision + Recall))`

Empirically, `β^2` is set to 0.3 to put more emphasis on precision.

Precision and Recall are calculated based on the binarized salient object mask and ground-truth:

`Precision` = `TP` / `TP` + `FP`, `Recall` = `TP` / `TP` + `FN`,

where `TP`, `TN`, `FP`, `FN` denote true-positive, true-negative, false-positive, and false-negative respectively.
More details regarding evaluation procedure can be found in this [paper](https://ieeexplore.ieee.org/document/5206596)

## Input

### Original model

Image, name - `input.1`, shape - `1, 3, 352, 352`, format `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order - `RGB`.
Mean values - [124.55, 118.90, 102.94]
Scale values - [56.77,  55.97,  57.50]

### Converted model

Image, name - `input.1`, shape - `1, 3, 352, 352`, format `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order - `BGR`.

## Output

### Original model
Saliency map, name `saliency_map`, shape `1, 1, 352, 352`, format `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

[Sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) should be applied on saliency map for conversion probability into [0, 1] range.

### Converted model

The converted model has the same parameters as the original model.

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
[license](https://github.com/weijun88/F3Net/blob/master/LICENSE):

```
MIT License

Copyright (c) 2019 Jun Wei

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
