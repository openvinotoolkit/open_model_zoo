# Brain-tumor-segmentation-0001

## Use Case and High-Level Description

This model was created for participation in the [Brain Tumor Segmentation Challenge](https://www.med.upenn.edu/sbia/brats2018.html) (BraTS) 2018.  
The model is based on [the corresponding paper](https://arxiv.org/abs/1810.04008), where authors present deep cascaded approach for automatic brain tumor segmentation. The paper describes modifications to 3D UNet architecture and specific augmentation strategy to efficiently handle multimodal MRI input. Besides this, the approach to enhance segmentation quality with context obtained from models of the same topology operating on downscaled data is introduced.  
Each input modality has its own encoder which are later fused together to produce single output segmentation.  

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Segmentation  |
| GFLOPs            | 409.996       |
| MParams           | 38.192        |
| Source framework  | MXNet         |

## Accuracy

See [https://github.com/lachinov/brats2018-graphlabunn]()


## Performance

## Input

The model takes as an input four MRI modalities `T1`, `T2`, `T1ce`, `Flair`. The inputs are cropped, resamped and z-score normalized. You can find additional information on the BraTS 2018 [page](https://www.med.upenn.edu/sbia/brats2018/data.html) and [wiki](https://en.wikipedia.org/wiki/Magnetic_resonance_imaging).  
In the preprocessing pipeline, all non-zero voxels are cropped and resampled to `128,128,128` resolution first. Then, each modality is z-score normalized separately. The input tensor is a concatenation of the four input modalities.

### Original model

MR Image, name - `data_crop`, shape - `1,4,128,128,128`, format is `B,C,D,H,W` where:

- `B` - batch size
- `C` - channel
- `D` - depth
- `H` - height
- `W` - width

The channels are ordered as `T1`, `T2`, `T1ce`, `Flair`.

### Converted model

MR Image, name - `data_crop`, shape - `1,4,128,128,128`, format is `B,C,D,H,W` where:

- `B` - batch size
- `C` - channel
- `D` - depth
- `H` - height
- `W` - width

The channels are ordered as `T1`, `T2`, `T1ce`, `Flair`.

## Output

### Original model


Probabilities of the given voxel to be in the corresponding class, name - `softmax_lbl3`, shape - `1,4,128,128,128`, output data format is `B,C,D,H,W` where:

- `B` - batch size
- `C` - channel
- `D` - depth
- `H` - height
- `W` - width

With the following channels: `background`, `necrotic core`, `edema` and `enhancing tumor`.

### Converted model

Probabilities of the given voxel to be in the corresponding class, name - `softmax_lbl3`, shape - `1,4,128,128,128`, output data format is `B,C,D,H,W` where:

- `B` - batch size
- `C` - channel
- `D` - depth
- `H` - height
- `W` - width

With the following channels: `background`, `necrotic core`, `edema` and `enhancing tumor`.

## Legal Information

[https://github.com/lachinov/brats2018-graphlabunn/blob/master/LICENSE]()
