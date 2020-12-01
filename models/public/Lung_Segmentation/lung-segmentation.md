# lung-cancer-screening(Multi-stage)

## Use Case and High-Level Description

This is a lung cancer screening composite model which segments the lung region in stage 1. In stage 2, the model detects the presence of nodules in patches extracted from the previously segmented lung region.

Reference: [Paper](https://arxiv.org/abs/2006.09308)
           [Dataset](https://luna16.grand-challenge.org/Data/)

## Multi-Stage model specification

| Metric                       | Value     |
|------------------------------|-----------|
| Dice coefficient for stage 1 | 0.979     |
| Accuracy for stage 2         | 98%       |
| Source framework             | PyTorch\* |


## Segmentation(stage1) model specification

lung-cancer-screening-stage1 model is a Convolutional Neural Network(CNN). 

The CNN is defined according to [UNet](https://arxiv.org/abs/1505.04597).

| Metric    | Value        |
|-----------|--------------|
| Type      | segmentation |
| GFlops    | 261.901      |
| MParams   | 34.526       |

### Input

#### Original Model

CT Image, name - 0, shape - `1,1,512,512`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

#### Converted Model

CT Image, name - 0, shape - `1,1,512,512`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

## Output

### Original Model

Probabilities of the given pixel to be in the corresponding class, name - 304, shape - 1,1,512,512, output data format is B,C,D,H,W, where:

B - batch size
C - channel
H - height
W - width

The channels are ordered as `background`,`lung`

### Converted Model

Probabilities of the given pixel to be in the corresponding class, name - 304, shape - 1,1,512,512, output data format is B,C,D,H,W, where:

B - batch size
C - channel
H - height
W - width

The channels are ordered as `background`,`lung`

<!-- ## Detection(stage2) model specification

| Metric  |     Value      |
|---------|----------------|
| GFlops  | 0.002796       |
| MParams | 0.050          |
| Type    | classification |

### Performance

### Input

#### Original model


#### Converted model


### Output

#### Original model

#### Converted model -->

## Legal Information

The original model is distributed under [Apache 2.0](https://drive.google.com/file/d/1LVVi_TUnIgR2Zl8OK_Jy2BNRJm9aD4iQ/view?usp=sharing)
