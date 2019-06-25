# mtcnn-r

## Use Case and High-Level Description

The `mtcnn-r` model is one of the [mtcnn](https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf) group of models designed to perform face detection. Short for "Multi-task Cascaded Convolutional Neural Network", it is implemented using the Caffe\* framework. The "r" designation indicates that this model is the "refine" network intended to refine the data returned as output from the "proposal" `mtcnn-p` network. For details about this family of models, check out the [repository](https://github.com/DuinoDu/mtcnn).

The model input is a blob with a vector containing the first pass of face data, as returned by the `mtcnn-p` model. The mean values need to be subtracted as follows: [127.5, 127.5, 127.5] before passing the image blob into the network. In addition, values must be divided by 0.0078125.

The model output is a blob with a vector containing the refined face data. If there are no faces detected by the refine pass, no further processing is needed. Otherwise, you will typically use this output as input to the `mtcnn-o` model.

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Detection     |
| GFLOPs            | 0.003         |
| MParams           | 0.1           |
| Source framework  | Caffe\*         |

## Accuracy

## Performance

## Input

### Original model

Image, name - `data`, shape - `1,3,24,24` in `B,C,W,H` format, where

* `B` - input batch size
* `C` - number of image channels
* `W` - width
* `H` - height

Expected color order: `RGB`
Mean values - [127.5, 127.5, 127.5], scale value - 128

### Converted model

Image, name - `data`, shape - `1,3,24,24` in `B,C,W,H` format, where

* `B` - input batch size
* `C` - number of image channels
* `W` - width
* `H` - height

Expected color order: `RGB`

## Output

### Original model

1. Face detection, name - `prob1`, shape - `1,2,B`, contains scores across two classes (`0 `- no face, `1` - face) for each input in batch. This is necessary to refine face regions from `mtcnn-p`.
2. Face location, name - `conv5-2`, contains clarifications for boxes produced by `mtcnn-p`.

### Converted model

1. Face detection, name - `prob1`, shape - `1,2,B`, contains scores across two classes (`0 `- no face, `1` - face) for each input in batch. This is necessary to refine face regions from `mtcnn-p`.
2. Face location, name - `conv5-2`, contains clarifications for boxes produced by `mtcnn-p`.

## Legal Information

[https://raw.githubusercontent.com/DuinoDu/mtcnn/master/LICENSE]()
