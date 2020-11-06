# common-sign-language-0001

## Use Case and High-Level Description

A human gesture recognition model for the Jester dataset recognition scenario
(gesture-level recognition). The model uses an S3D framework with MobileNet V3 backbone.
Please refer to the [Jester\*](https://20bn.com/datasets/jester) dataset specification
to see the list of gestures that are recognized by this model.

The model accepts a stack of frames (8 frames) sampled with a constant framerate (15 FPS)
and produces a prediction on the input clip.

## Specification

| Metric                                  | Value        |
|-----------------------------------------|--------------|
| Top-1 accuracy (continuous Jester\*)    | 93.58%       |
| GFlops                                  | 4.2269       |
| MParams                                 | 4.1128       |
| Source framework                        | PyTorch\*    |

## Input

### Original Model

Batch of images of the shape [1x3x8x224x224] in the [BxCxTxHxW] format, where:

- `B` - batch size
- `C` - channel
- `T` - sequence length
- `H` - height
- `W` - width

Channel order is `RGB`.

### Converted Model

Batch of images of the shape [1x3x8x224x224] in the [BxCxTxHxW] format, where:

- `B` - batch size
- `C` - channel
- `T` - sequence length
- `H` - height
- `W` - width

Channel order is `RGB`.

## Output

The model outputs a tensor with the shape [Bx27], each row is a logits vector of performed Jester\* gestures.

### Original Model

Blob of the shape [1, 27] in the [BxC] format, where:

- `B` - batch size
- `C` - predicted logits size

### Converted Model

Blob of the shape [1, 27] in the [BxC] format, where:

- `B` - batch size
- `C` - predicted logits size


## Legal Information
[\*] Other names and brands may be claimed as the property of others.

The original model is distributed under the
[Apache License 2.0](https://github.com/sovrasov/mmaction2/blob/ote/LICENSE).
