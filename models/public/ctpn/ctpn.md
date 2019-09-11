# ctpn

## Use Case and High-Level Description

Detecting Text in Natural Image with Connectionist Text Proposal Network. For details see [paper](https://arxiv.org/pdf/1609.03605.pdf).

## Example

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Object detection                          |
| GFlops                          | 55.813                                    |
| MParams                         | 17.237                                    |
| Source framework                | Tensorflow\*                              |

## Performance

## Input

### Original Model

Image, name: `image_tensor`, shape: [1x600x600x3], format: [BxHxWxC],
   where:

    - B - batch size
    - H - image height
    - W - image width
    - C - number of channels

   Expected color order: BGR.
   Mean values: [102.9801, 115.9465, 122.7717].

### Converted Model

Image, name: `Placeholder`, shape: [1x3x600x600], format: [BxCxHxW],
   where:

    - B - batch size
    - C - number of channels
    - H - image height
    - W - image width

   Expected color order: BGR.

## Output

### Original Model

1. Detection boxes, name: `rpn_bbox_pred/Reshape_1`, contains predicted regions, in format [BxHxWxA], where:

    - B - batch size
    - H - image height
    - W - image width
    - A - vector of 4\*N coordinates, where N is the number of detected anchors.

2. Probability, name: `Reshape_2`, contains probabilities for predicted regions in a [0,1] range in format [BxHxWxA], where:

    - B - batch size
    - H - image height
    - W - image width
    - A - vector of 4\*N coordinates, where N is the number of detected anchors.

### Converted Model

1. Detection boxes, name: `rpn_bbox_pred/Reshape_1/Transpose`, shape: [1x40x18x18] contains predicted regions, format: [BxAxHxW], where:

    - B - batch size
    - A - vector of 4\*N coordinates, where N is the number of detected anchors.
    - H - image height
    - W - image width

2. Probability, name: `Reshape_2/Transpose`, shape: [1x20x18x18], contains probabilities for predicted regions in a[0,1] range in format [BxAxHxW], where:

    - B - batch size
    - A - vector of 2\*N class probabilities (0 class for background, 1 class for text), where N is the number of detected anchors.
    - H - image height
    - W - image width

## Legal Information

[https://raw.githubusercontent.com/eragonruan/text-detection-ctpn/banjin-dev/LICENSE]()
