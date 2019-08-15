# colorization-v2-norebal

## Use Case and High-Level Description

The `colorization-v2-norebal` model is one of the [colorization](https://arxiv.org/pdf/1603.08511)
group of models designed to perform image colorization. For details
about this family of models, check out the [repository](https://github.com/richzhang/colorization).

The gray-scale or BGR image should be preprocessed first:
1) divide image's values by 255
2) convert the image to LAB
3) extract l component with 50.0 subtracted from each value

The model input is a blob that consists of a L-channel of image of 1x1x224x224. 
The model output is a blob that consists of AB channels of image of 1x2x56x56. 

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Colorization  |
| GFLOPs            | -             |
| MParams           | -             |
| Source framework  | Caffe\*       |

## Accuracy

## Performance

## Input

### Original model

Image, name - `data_l`,  shape - `1,1,224,224`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is L-channel. 
Mean values - 50, scale value - 255

### Converted model

Image, name - `data_l`,  shape - `1,1,224,224`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is L-channel. 

## Output


### Original model

Image, name - `class8_ab`\*,  shape - `1,2,56,56`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

### Converted model

Image, name - `class8_313_rh`\*,  shape - `1,313,56,56`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

> **NOTE**: `class8_313_rh` layer is in front of `class8_ab` layer, 
in order for network to work, 
you need to reproduce `class8_ab` layer with the coefficients that 
downloaded separately with the model.

## Legal Information

[https://raw.githubusercontent.com/richzhang/colorization/master/LICENSE]()
