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

The input is cropped, resamped and z-score normalzied image.  
Image, shape - `1,4,128,128,128`, format is `B,C,D,H,W` where:

- `B` - batch size
- `C` - channel
- `D` - depth
- `H` - height
- `W` - width

The channels are ordered as `T1`, `T2`, `T1ce`, `Flair`.

## Output

Tensor of shape `1,4,128,128,128` with probabilities of the given voxel to be in the `background`, `necrotic core`, `edema` or `enhancing tumor` class respectively.

## Legal Information

[https://github.com/lachinov/brats2018-graphlabunn/blob/master/LICENSE]()
