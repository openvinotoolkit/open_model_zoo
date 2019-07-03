# densenet-169

## Use Case and High-Level Description

The `densenet-169` model is one of the [DenseNet](https://arxiv.org/pdf/1608.06993)
group of models designed to perform image classification. The main difference with
the `densenet-121` model is the size and accuracy of the model. The `densenet-169`
is larger at just about 55MB in size vs the `densenet-121` model's roughly 31MB size.
Originally trained on Torch, the authors converted them into Caffe* format. All
the DenseNet models have been pretrained on the ImageNet image database. For details
about this family of models, check out the [repository](https://github.com/shicai/DenseNet-Caffe).

The model input is a blob that consists of a single image of 1x3x224x224 in BGR
order. The BGR mean values need to be subtracted as follows: [103.94, 116.78, 123.68]
before passing the image blob into the network. In addition, values must be divided
by 0.017.

The model output for `densenet-169` is the typical object classifier output for
the 1000 different classifications matching those in the ImageNet database.

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 6.788         |
| MParams           | 14.139        |
| Source framework  | Caffe\*         |

## Accuracy

See [https://github.com/shicai/DenseNet-Caffe]().

## Performance

## Input

### Original model

Image, name - `data`,  shape - `1,3,224,224`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.
Mean values - [103.94,116.78,123.68], scale value - 58.8235294117647

### Converted model

Image, name - `data`,  shape - `1,3,224,224`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`

## Output

### Original model

Object classifier according to ImageNet classes, name - `prob`,  shape - `1,1000,1,1`, contains predicted
probability for each class in logits format

### Converted model

Object classifier according to ImageNet classes, name - `prob`,  shape - `1,1000,1,1`, contains predicted
probability for each class in logits format

## License

[https://raw.githubusercontent.com/liuzhuang13/DenseNet/master/LICENSE]()
