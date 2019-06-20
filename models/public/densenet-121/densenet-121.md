# densenet-121

## Use Case and High-Level Description

The `densenet-121` model is one of the [DenseNet](https://arxiv.org/pdf/1608.06993)
group of models designed to perform image classification. Originally trained on
Torch, the authors converted them into Caffe* format. All the DenseNet models have
been pretrained on the ImageNet image database. For details about this family of
models, check out the [repository](https://github.com/shicai/DenseNet-Caffe). 

The model input is a blob that consists of a single image of 1x3x224x224 in BGR
order. The BGR mean values need to be subtracted as follows: [103.94, 116.78, 123.68]
before passing the image blob into the network. In addition, values must be scaled
by 0.017.

The model output for `densenet-121` is the typical object classifier output for
the 1000 different classifications matching those in the ImageNet database.

## Example

## Specification

## Accuracy

See [https://github.com/shicai/DenseNet-Caffe]()

## Performance

## Inputs

Name - `data`, shape - `1,3,224,224`, image format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`
 
## Outputs

Name: `fc6`

## License

[https://raw.githubusercontent.com/liuzhuang13/DenseNet/master/LICENSE]()
