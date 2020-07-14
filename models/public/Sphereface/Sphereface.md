# Sphereface

## Use Case and High-Level Description

[Deep face recognition under open-set protocol](https://arxiv.org/abs/1704.08063)

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Face recognition |
| GFLOPs            | 3.504         |
| MParams           | 22.671        |
| Source framework  | Caffe\*       |

## Accuracy

| Metric | Value |
| ------ | ----- |
| LFW accuracy | 98.8321%|

## Performance

## Input

### Original model

Image, name - `data`,  shape - `1,3,112,96`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.
Mean values - [127.5,127.5,127.5], scale value - 128

### Converted model

Image, name - `data`,  shape - `1,3,112,96`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original model

Face embeddings, name - `fc5`,  shape - `1,512`, output data format  - `B,C`, where:

- `B` - batch size
- `C` - row-vector of 512 floating points values, face embeddings

The net outputs on different images are comparable in cosine distance.

### Converted model

Face embeddings, name - `fc5`,  shape - `1,512`, output data format  - `B,C`, where:

- `B` - batch size
- `C` - row-vector of 512 floating points values, face embeddings

The net outputs on different images are comparable in cosine distance.

## Legal Information

The original model is distributed under the following
[license](https://raw.githubusercontent.com/wy1iu/sphereface/master/LICENSE):

```
MIT License

Copyright (c) 2017 Weiyang Liu and Yandong Wen

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
