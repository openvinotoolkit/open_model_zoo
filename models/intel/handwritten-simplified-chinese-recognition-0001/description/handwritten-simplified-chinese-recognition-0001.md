# handwritten-simplified-chinese-recognition-0001

## Use Case and High-Level Description

This is a network for handwritten simplified Chinese text recognition scenario. It consists of a VGG16-like backbone,
reshape layer and a fully connected layer.
The network is able to recognize simplified Chinese text consisting of characters in the [SCUT-EPT](https://github.com/HCIILAB/SCUT-EPT_Dataset_Release).

## Example

![](./test.png) -> 的人不一了是他有为在责新中任自之我们

## Specification

| Metric                                         | Value              |
|------------------------------------------------|--------------------|
| GFlops                                         |                    |
| MParams                                        |                    |
| Accuracy on partial SCUT-EPT test set          |   75.58%           |
| Source framework                               | PyTorch\*          |


## Accuracy Values

This demo adopts [label error rate](https://dl.acm.org/doi/abs/10.1145/1143844.1143891) as the metric for accuracy.

## Inputs
Shape: [1x1x96x2000] - An input image in the format [BxCxHxW],
where:
  - B - batch size
  - C - number of channels
  - H - image height
  - W - image width

Note that the source image will be converted to grayscale, resized to specific height (such as 96) while keeping aspect ratio, and right-bottom padded.

## Outputs

The net outputs a blob with the shape [125x1x4059] in the format [WxBxL], where:
  - W - output sequence length
  - B - batch size
  - L - confidence distribution across the supported symbols in [SCUT-EPT](https://github.com/HCIILAB/SCUT-EPT_Dataset_Release).

The network output can be decoded by CTC Greedy Decoder.

## Legal Information
[*] Other names and brands may be claimed as the property of others.
