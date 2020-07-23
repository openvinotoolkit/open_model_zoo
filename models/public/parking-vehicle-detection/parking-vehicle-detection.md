# Cascade-RCNN

## Use Case and High-Level Description
  The "Cascade-RCNN" model is two-stage multibox Detection network intended to perform 
  object detection. This model is implemented using mmdetection framework.

  The model input is a blob that consists of a single image of 1x3x800x1333 in BGR
  order. The BGR mean values need to be subtracted
  as follows: [123.675, 116.28, 103.53] before passing the image blob into the network. 
  Also BGR values need to be scaled by division on following coefficients: [58.395, 57.12, 57.375]  


## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Detection     |
| GFLOPs            | 234.46        |
| MParams           |   tbd         |
| Source framework  | Pytorch\*     |

## Accuracy

TBD

## Performance

## Input

### Original model

Image, name - `image`,  shape - `1,3,800,1333`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.
Mean values - [123.675, 116.28, 103.53], scale value -  [58.395, 57.12, 57.375]

## Output

### Original model 

The array of detection summary info, names - `boxes, labels`,  shape - `[1, 1, N, 5], [N,1]`, where N is the number of detected bounding boxes. For each detection, the description has the format:
[`x_min`, `y_min`, `x_max`, `y_max`, `conf`], where:

- `conf` - confidence for the predicted class
- (`x_min`, `y_min`) - coordinates of the top left bounding box corner (coordinates are in normalized format, in range [0, 1])
- (`x_max`, `y_max`) - coordinates of the bottom right bounding box corner  (coordinates are in normalized format, in range [0, 1])
and 
[`label`] - predicted class (vehicle)

## Legal Information
[*] Other names and brands may be claimed as the property of others.
```
MIT License

Copyright (c) 2018 chuanqi305

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
