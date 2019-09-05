# densenet-121

## Use Case and High-Level Description

The `densenet-121` model is one of the [DenseNet*](https://arxiv.org/pdf/1608.06993)
group of models designed to perform image classification. The authors originally trained the models 
on Torch\*, but then converted them into Caffe\* format. All DenseNet models have
been pretrained on the ImageNet image database. For details about this family of
models, check out the [repository](https://github.com/shicai/DenseNet-Caffe).


## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 5.724         |
| MParams           | 7.971         |
| Source framework  | Caffe\*         |

## Accuracy

See [https://github.com/shicai/DenseNet-Caffe]()

## Performance

## Input

The model input is a blob that consists of a single image of 1x3x224x224 in BGR
order. Before passing the image blob into the network, subtract BGR mean values 
as follows: [103.94, 116.78, 123.68]. In addition, values must be divided by 0.017.

### Original Model

Image, name - `data`,  shape - `1,3,224,224`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`. 
Mean values - [103.94,116.78,123.68], scale value - 58.8235294117647

### Converted Model

Image, name - `data`,  shape - `1,3,224,224`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

The model output for `densenet-121` is a typical object classifier output for 1000 different 
classifications matching those in the ImageNet database.

### Original Model

Object classifier according to ImageNet classes, name - `prob`,  shape - `1,1000,1,1`, contains predicted
probability for each class in logits format.

### Converted Model

Object classifier according to ImageNet classes, name - `prob`,  shape - `1,1000,1,1`, contains predicted
probability for each class in logits format.

## Legal Information

[https://raw.githubusercontent.com/liuzhuang13/DenseNet/master/LICENSE]()
