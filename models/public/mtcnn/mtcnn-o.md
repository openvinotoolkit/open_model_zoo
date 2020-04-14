# mtcnn-o

## Use Case and High-Level Description

The `mtcnn-o` model is the third of the [mtcnn](https://arxiv.org/abs/1604.02878) group of models designed to perform face detection. Short for "Multi-task Cascaded Convolutional Neural Network", it is implemented using the Caffe\* framework. The "o" designation indicates that this model is the "output" network intended to take the data returned from the "refine" `mtcnn-r` network, and transform it into the final output data.  For details about this family of models, check out the [repository](https://github.com/DuinoDu/mtcnn).

The model input is a blob with a vector containing the refined face data, as returned by the `mtcnn-r` model. The mean values need to be subtracted as follows: [127.5, 127.5, 127.5] before passing the image blob into the network. In addition, values must be divided by 0.0078125.

The model output is a blob with a vector containing the output face data.

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Detection     |
| GFLOPs            | 0.026         |
| MParams           | 0.389         |
| Source framework  | Caffe\*       |

## Accuracy

## Performance

## Input

### Original model

Image, name - `data`, shape - `1,3,48,48` in `B,C,W,H` format, where

- `B` - input batch size
- `C` - number of image channels
- `W` - width
- `H` - height

Expected color order: `RGB`.
Mean values - [127.5, 127.5, 127.5], scale value - 128

### Converted model

Image, name - `data`, shape - `1,3,48,48` in `B,C,W,H` format, where

- `B` - input batch size
- `C` - number of image channels
- `W` - width
- `H` - height

Expected color order: `RGB`.

## Output

### Original model

1. Face detection, name - `prob1`, shape  - `1,2,B`, contains scores across two classes (`0 `- no face, `1` - face) for each input in batch. This is necessary for final face regions refining after`mtcnn-p` and `mtcnn-r`.
2. Face location, name - `conv6-2`, contains final clarifications for boxes produced by `mtcnn-p` and refined by `mtcnn-r`.
3. Control points, name - `conv6-3`, contains five facial landmarks: `left eye`, `right eye`, `nose`, `left mouth corner`, `right mouth corner` coordinates for each face region.

### Converted model

1. Face detection, name - `prob1`, shape  - `1,2,B`, contains scores across two classes (`0 `- no face, `1` - face) for each input in batch. This is necessary for final face regions refining after`mtcnn-p` and `mtcnn-r`.
2. Face location, name - `conv6-2`, contains final clarifications for boxes produced by `mtcnn-p` and refined by `mtcnn-r`.
3. Control points, name - `conv6-3`, contains five facial landmarks: `left eye`, `right eye`, `nose`, `left mouth corner`, `right mouth corner` coordinates for each face region.

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
