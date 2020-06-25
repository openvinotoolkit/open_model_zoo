# mtcnn (composite)

## Use Case and High-Level Description

The composite `mtcnn` model is [mtcnn](https://arxiv.org/abs/1604.02878) group of models designed to perform face detection. Short for "Multi-task Cascaded Convolutional Neural Network", it is implemented using the Caffe\* framework.  For details about this family of models, check out the [repository](https://github.com/DuinoDu/mtcnn).

The input for each models is a blob with specific face data. The mean values need to be subtracted as follows: [127.5, 127.5, 127.5] before passing the image blob into the network. In addition, values must be divided by 128.

## Composite model specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Detection     |
| Source framework  | Caffe\*       |

## Accuracy

| Metric | Value |
| ------ | ----- |
| mAP | 48.1308%|
| Recall | 62.2625%|

## mtcnn-p model specification

The "p" designation indicates that the `mtcnn-p` model is the "proposal" network intended to find the initial set of faces.

The model input is an image containing the data to be analyzed.

The model output is a blob with a vector containing the first pass of face data. If there are no faces detected, no further processing is needed. Otherwise, you will typically use this output as input to the `mtcnn-r` model.

| Metric            | Value         |
|-------------------|---------------|
| GFLOPs            | 3.366         |
| MParams           | 0.007         |


### Performance

### Input

#### Original model

Image, shape - `1,3,720,1280`, format is `B,C,W,H`, where:

- `B` - batch size
- `C` - channel
- `W` - width
- `H` - height

Expected color order: `RGB`.
Mean values - [127.5, 127.5, 127.5], scale value - 128

#### Converted model

Image, shape - `1,3,720,1280`, format is `B,C,W,H`, where:

- `B` - batch size
- `C` - channel
- `W` - width
- `H` - height

Expected color order: `RGB`.

### Output

#### Original model

1. Face detection, name - `prob1`, shape - `1,2,W,H`, contains scores across two classes (0 - no face, 1 - face) for each pixel whether it contains face or not.
2. Face location, name - `conv4-2`, contains regions with detected faces.

#### Converted model

1. Face detection, name - `prob1`, shape - `1,2,W,H`, contains scores across two classes (0 - no face, 1 - face) for each pixel whether it contains face or not.
2. Face location, name - `conv4-2`, contains regions with detected faces.


## mtcnn-r model specification

The "r" designation indicates that the `mtcnn-r` model is the "refine" network intended to refine the data returned as output from the "proposal" `mtcnn-p` network.

The model input is a blob with a vector containing the first pass of face data, as returned by the `mtcnn-p` model.

The model output is a blob with a vector containing the refined face data. If there are no faces detected by the refine pass, no further processing is needed. Otherwise, you will typically use this output as input to the `mtcnn-o` model.

| Metric            | Value         |
|-------------------|---------------|
| GFLOPs            | 0.003         |
| MParams           | 0.1           |


### Performance

### Input

#### Original model

Image, name - `data`, shape - `1,3,24,24` in `B,C,W,H` format, where

* `B` - input batch size
* `C` - number of image channels
* `W` - width
* `H` - height

Expected color order: `RGB`
Mean values - [127.5, 127.5, 127.5], scale value - 128

#### Converted model

Image, name - `data`, shape - `1,3,24,24` in `B,C,W,H` format, where

* `B` - input batch size
* `C` - number of image channels
* `W` - width
* `H` - height

Expected color order: `RGB`

### Output

#### Original model

1. Face detection, name - `prob1`, shape - `1,2,B`, contains scores across two classes (`0 `- no face, `1` - face) for each input in batch. This is necessary to refine face regions from `mtcnn-p`.
2. Face location, name - `conv5-2`, contains clarifications for boxes produced by `mtcnn-p`.

#### Converted model

1. Face detection, name - `prob1`, shape - `1,2,B`, contains scores across two classes (`0 `- no face, `1` - face) for each input in batch. This is necessary to refine face regions from `mtcnn-p`.
2. Face location, name - `conv5-2`, contains clarifications for boxes produced by `mtcnn-p`.


## mtcnn-o model specification

The "o" designation indicates that the `mtcnn-o` model is the "output" network intended to take the data returned from the "refine" `mtcnn-r` network, and transform it into the final output data.

The model input is a blob with a vector containing the refined face data, as returned by the `mtcnn-r` model.

The model output is a blob with a vector containing the output face data.

| Metric            | Value         |
|-------------------|---------------|
| GFLOPs            | 0.026         |
| MParams           | 0.389         |


### Performance

### Input

#### Original model

Image, name - `data`, shape - `1,3,48,48` in `B,C,W,H` format, where

- `B` - input batch size
- `C` - number of image channels
- `W` - width
- `H` - height

Expected color order: `RGB`.
Mean values - [127.5, 127.5, 127.5], scale value - 128

#### Converted model

Image, name - `data`, shape - `1,3,48,48` in `B,C,W,H` format, where

- `B` - input batch size
- `C` - number of image channels
- `W` - width
- `H` - height

Expected color order: `RGB`.

### Output

#### Original model

1. Face detection, name - `prob1`, shape  - `1,2,B`, contains scores across two classes (`0 `- no face, `1` - face) for each input in batch. This is necessary for final face regions refining after`mtcnn-p` and `mtcnn-r`.
2. Face location, name - `conv6-2`, contains final clarifications for boxes produced by `mtcnn-p` and refined by `mtcnn-r`.
3. Control points, name - `conv6-3`, contains five facial landmarks: `left eye`, `right eye`, `nose`, `left mouth corner`, `right mouth corner` coordinates for each face region.

#### Converted model

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
