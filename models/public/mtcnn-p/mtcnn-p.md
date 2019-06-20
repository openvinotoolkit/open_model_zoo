# mtcnn-p

## Use Case and High-Level Description

The `mtcnn-p` model is one of the [mtcnn](https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf) group of models designed to perform face detection. Short for "Multi-task Cascaded Convolutional Neural Network", it is implemented using the Caffe framework. The "p" designation indicates that this model is the "proposal" network intended to find the initial set of faces. For details about this family of models, check out the [repository](https://github.com/DuinoDu/mtcnn).

The model input is an image containing the data to be analyzed.

The model output is a blob with a vector containing the first pass of face data. If there are no faces detected, no further processing is needed. Otherwise, you will typically use this output as input to the `mtcnn-r` model.

## Example

## Specification

## Accuracy

## Performance

## Inputs

Name - `data`, shape - `1,3,720,1280`, image format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

## Outputs

Name: `prob1`

## Legal Information

[https://raw.githubusercontent.com/DuinoDu/mtcnn/master/LICENSE]()
