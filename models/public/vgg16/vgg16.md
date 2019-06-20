# vgg16

## Use Case and High-Level Description

The `vgg16` model is one of the [vgg](https://arxiv.org/pdf/1409.1556.pdf) models designed to perform image classification in Caffe format.

The model input is a blob that consists of a single image of "1x3x224x224" in BGR order. The BGR mean values need to be subtracted as follows: [103.939, 116.779, 123.68] before passing the image blob into the network.

The model output for `vgg16` is the typical object classifier output for the 1000 different classifications matching those in the ImageNet database.

## Example

## Specification

## Accuracy

## Performance

## Inputs

Name - `data`, shape - `1,3,224,224`

## Outputs

Name: `prob`

## Legal Information

[https://raw.githubusercontent.com/keras-team/keras/master/LICENSE]()
