# mosaic

## Use Case and High-Level Description

The `mosaic` model is one of the style transfer models designed to mix the content of an image with the style of another image. The model uses the method described in [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) along with [Instance Normalization](https://arxiv.org/pdf/1607.08022.pdf).


## Example


## Specification

| Metric            | Value            |
|-------------------|------------------|
| Type              | Style Transfer   |
| GFLOPs            | -                |
| MParams           | -                |
| Source framework  | ONNX\*           |

## Accuracy

Accuracy metrics are obtained on MS COCO val2017 dataset. Images were resized to input size.

| Metric | Original model | Converted model (FP32) | Converted model (FP16) |
| ------ | -------------- | ---------------------- | ---------------------- |
| PSNR   | 12.03Db        | 12.03Db                | 12.04Db                |

## Performance

## Input

### Original model

Image, name - `input1`,  shape - `1,3,224,224`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Expected color order: RGB.

### Converted model

Image, name - `input1`,  shape - `1,3,224,224`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Expected color order: BGR.

## Output

### Original model

NumPy float32 array, name - `output1`, shape - `1,3,224,224`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

### Converted model

NumPy float32 array, name - `output1`, shape - `1,3,224,224`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width



## Legal Information
The original model is distributed under the following
[license](https://raw.githubusercontent.com/onnx/models/master/LICENSE):

MIT License

```
Copyright (c) ONNX Project Contributors

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