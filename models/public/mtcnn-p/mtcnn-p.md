# mtcnn-p

## Use Case and High-Level Description

The `mtcnn-p` model is one of the [mtcnn](https://arxiv.org/abs/1604.02878) group of models designed to perform face detection. Short for "Multi-task Cascaded Convolutional Neural Network", it is implemented using the Caffe\* framework. The "p" designation indicates that this model is the "proposal" network intended to find the initial set of faces. For details about this family of models, check out the [repository](https://github.com/DuinoDu/mtcnn).

The model input is an image containing the data to be analyzed. The mean values need to be subtracted as follows: [127.5, 127.5, 127.5] before passing the image blob into the network. In addition, values must be divided by 0.0078125.

The model output is a blob with a vector containing the first pass of face data. If there are no faces detected, no further processing is needed. Otherwise, you will typically use this output as input to the `mtcnn-r` model.

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Detection     |
| GFLOPs            | 3.366         |
| MParams           | 0.007         |
| Source framework  | Caffe\*       |

## Accuracy

## Performance

## Input

### Original model

Image, shape - `1,3,720,1280`, format is `B,C,W,H`, where:

- `B` - batch size
- `C` - channel
- `W` - width
- `H` - height

Expected color order: `RGB`.
Mean values - [127.5, 127.5, 127.5], scale value - 128

### Converted model

Image, shape - `1,3,720,1280`, format is `B,C,W,H`, where:

- `B` - batch size
- `C` - channel
- `W` - width
- `H` - height

Expected color order: `RGB`.

## Output

### Original model

1. Face detection, name - `prob1`, shape - `1,2,W,H`, contains scores across two classes (0 - no face, 1 - face) for each pixel whether it contains face or not.
2. Face location, name - `conv4-2`, contains regions with detected faces.

### Converted model

1. Face detection, name - `prob1`, shape - `1,2,W,H`, contains scores across two classes (0 - no face, 1 - face) for each pixel whether it contains face or not.
2. Face location, name - `conv4-2`, contains regions with detected faces.

## Legal Information

The original model is distributed under the following
[license](https://raw.githubusercontent.com/DuinoDu/mtcnn/master/LICENSE):

```
MIT License

Copyright (c) 2016 Kaipeng Zhang

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
