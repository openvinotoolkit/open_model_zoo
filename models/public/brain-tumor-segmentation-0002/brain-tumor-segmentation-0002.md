# brain-tumor-segmentation-0002

## Use Case and High-Level Description

This model was created for participation in the [Brain Tumor Segmentation Challenge](https://www.med.upenn.edu/cbica/brats2019/registration.html) (BraTS) 2019. It has the UNet architecture trained with residual blocks.

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Segmentation  |
| GFLOPs            | 300.801       |
| MParams           | 4.51          |
| Source framework  | PyTorch\*       |

## Accuracy

See [BRATS 2019 Leaderboard](https://www.cbica.upenn.edu/BraTS19/lboardValidation.html). The metrics
for challenge validation (Dice_WT, Dice_TC, Dice_ET) differ from the metrics reported below (which
are compartible with input labels):

- WT (whole tumor) class combines all three tumor classes:
    - necrotic core / non-enhancing tumor
    - edema
    - enhancing tumor
- TC (tumor core) combines the following classes:
    - necrotic core
    - non-enhancing tumor
- ET (enhancing tumor)

The following accuracy metrics are measured on a `brain tumor` training subset of the [Medical Decathlon](http://medicaldecathlon.com/) dataset.

**Mean**:
- Dice index for "overall": 91.5%
- Dice index for "necrotic core / non-enhancing tumor": 61.1%
- Dice index for "edema": 80.6%
- Dice index for "enhancing tumor": 79.4%

**Median**:
- Dice index for "overall": 92.7%
- Dice index for "necrotic core / non-enhancing tumor": 64.5%
- Dice index for "edema": 83.5%
- Dice index for "enhancing tumor": 86%


> **NOTE**: The accuracy achieved with ONNX\* model adapted for OpenVINO™ can slightly differ from the accuracy achieved with the original PyTorch model since the upsampling operation was changed from the `trilinear` to `nearest` mode.

## Performance

## Input

The model takes as an input four MRI modalities `T1`, `T1ce`, `T2`, `Flair`. Find additional information on the [BraTS 2019 page](https://www.med.upenn.edu/cbica/brats2019/registration.html) and [wiki](https://en.wikipedia.org/wiki/Magnetic_resonance_imaging).
In the preprocessing pipeline, each modality should be z-score normalized separately. The input tensor is a concatenation of the four input modalities.

### Original Model

MR Image, name - `0`, shape - `1,4,128,128,128`, format is `B,C,D,H,W`, where:

- `B` - batch size
- `C` - channel
- `D` - depth
- `H` - height
- `W` - width

The channels are ordered as `T1`, `T1ce`, `T2`, `Flair`.

### Converted Model

MR Image, name - `0`, shape - `1,4,128,128,128`, format is `B,C,D,H,W`, where:

- `B` - batch size
- `C` - channel
- `D` - depth
- `H` - height
- `W` - width

The channels are ordered as `T1`, `T1ce`, `T2`, `Flair`.

## Output

### Original Model

Probabilities of the given voxel to be in the corresponding class, name - `304`, shape - `1,3,128,128,128`, output data format is `B,C,D,H,W`, where:

- `B` - batch size
- `C` - channel
- `D` - depth
- `H` - height
- `W` - width

The channels are ordered as `whole tumor`, `tumor core`, and `enhancing tumor`.

### Converted Model

Probabilities of the given voxel to be in the corresponding class, name - `304`, shape - `1,3,128,128,128`, output data format is `B,C,D,H,W`, where:

- `B` - batch size
- `C` - channel
- `D` - depth
- `H` - height
- `W` - width

The channels are ordered as `whole tumor`, `tumor core`, and `enhancing tumor`.

## Legal Information

The original model is distributed under the
[MIT License](https://raw.githubusercontent.com/lachinov/brats2019/master/LICENSE).

```
The MIT License

Copyright (c) 2019 Dmitrii Lachinov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
