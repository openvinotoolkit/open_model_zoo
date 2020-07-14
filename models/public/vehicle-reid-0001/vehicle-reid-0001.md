# vehicle-reid-0001

## Use Case and High-Level Description

This is a vehicle reidentification model for a general scenario. It uses a whole
car body image as an input and outputs an embedding vector to match a pair of images
by the cosine distance. The model is based on the OmniScaleNet backbone developed for fast inference.
A single reidentification head from the 1/16 scale
feature map outputs an embedding vector of 512 floats.

## Specification

| Metric                            | Value                                     |
|-----------------------------------|-------------------------------------------|
| VeRi-776\* rank-1                 | 96.31 %                                   |
| VeRi-776\* mAP                    | 85.15 %                                   |
| Camera location                   | All traffic cameras                       |
| Support of occluded vehicles      | YES                                       |
| Occlusion coverage                | <50%                                      |
| GFlops                            | 2.643                                     |
| MParams                           | 2.183                                     |
| Source framework                  | PyTorch\*                                 |

The cumulative matching curve (CMC) at rank-1 is accuracy denoting the possibility
to locate at least one true positive in the top-1 rank.
Mean Average Precision (mAP) is the mean across Average Precision (AP) of all queries.
AP is defined as the area under the
[precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall) curve.

## Performance

## Input

### Original Model

One image of the shape [1x3x208x208] in the [BxCxHxW] format, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `RGB`.

### Converted Model

One image of the shape [1x3x208x208] in the [BxCxHxW] format, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

The net outputs a vector decriptor, which can be compared with other descriptors using the
[cosine distance](https://en.wikipedia.org/wiki/Cosine_similarity).

### Original Model

Blob of the shape [1, 512] in the [BxC] format, where:

- `B` - batch size
- `C` - predicted descriptor size

### Converted Model

Blob of the shape [1, 512] in the [BxC] format, where:

- `B` - batch size
- `C` - predicted descriptor size


## Legal Information
[\*] Other names and brands may be claimed as the property of others.

The original model is distributed under the
[MIT License](https://raw.githubusercontent.com/sovrasov/deep-person-reid/vehicle_reid/LICENSE).

```
MIT License

Copyright (c) 2018 Kaiyang Zhou

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
