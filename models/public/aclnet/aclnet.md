# AclNet

## Use Case and High-Level Description

The `AclNet` model is designed to perform sound classification.
The `AclNet` model is trained on a dataset of environmental sounds (DES-53).
For details about the model, see this [paper](https://arxiv.org/pdf/1811.06669.pdf).

The model input is a segment of PCM audio samples in [N, C, H, W] format with the shape [1, 1, 1, 16000].

The model output for `AclNet` is the sound classifier output for the 53 different environmental sound classes from the DES-53 database.

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| MParams           | 2.7           |
| Source framework  | PyTorch\*     |

## Accuracy

See this [publication](http://dcase.community/documents/workshop2019/proceedings/DCASE2019Workshop_Huang_52.pdf) and this [paper](https://arxiv.org/pdf/1811.06669.pdf).

## Performance

## Input

Audio, name - `data`, shape - `1,1,1,16000`, format is `N,C,H,W` where:

- `N` - number of samples
- `C` - channel
- `H` - height
- `W` - width

## Output

Sound classifier according to DES-53 classes, name - `softmax`, shape - `1,53`, output data format is `N,C` where:

- `N` - number of samples classified
- `C` - Predicted softmax scores for each class in [0, 1] range

## Legal Information

The original model is distributed under [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0.html)
