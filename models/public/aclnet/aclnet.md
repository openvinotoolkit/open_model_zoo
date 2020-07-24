# aclnet

## Use Case and High-Level Description

The `AclNet` model is designed to perform sound classification.
The `AclNet` model is trained on a dataset of environmental sounds (DES-53).
For details about the model, see this [paper](https://arxiv.org/pdf/1811.06669.pdf).

The model input is a segment of PCM audio samples in [N, C, 1, L] format.

The model output for `AclNet` is the sound classifier output for the 53 different environmental sound classes from the DES-53 database.

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 1.4           |
| MParams           | 2.7           |
| Source framework  | PyTorch\*     |

## Accuracy

See this [publication](http://dcase.community/documents/workshop2019/proceedings/DCASE2019Workshop_Huang_52.pdf) and this [paper](https://arxiv.org/pdf/1811.06669.pdf).

## Performance

## Input

### Original Model

Audio, name - `0`, shape - `1,1,1,L`, format is `N,C,1,L` where:

- `N` - batch size
- `C` - channel
- `L` - number of PCM samples (minimum value is 16000)

### Converted Model

Audio, name - `0`, shape - `1,1,1,L`, format is `N,C,1,L` where:

- `N` - batch size
- `C` - channel
- `L` - number of PCM samples (minimum value is 16000)

## Output

### Original Model

Sound classifier according to DES-53 classes, name - `203`, shape - `1,53`, output data format is `N,C` where:

- `N` - batch size
- `C` - Predicted softmax scores for each class in [0, 1] range

### Converted Model

Sound classifier according to DES-53 classes, name - `203`, shape - `1,53`, output data format is `N,C` where:

- `N` - batch size
- `C` - Predicted softmax scores for each class in [0, 1] range

## Legal Information

The original model is distributed under [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0.html).
