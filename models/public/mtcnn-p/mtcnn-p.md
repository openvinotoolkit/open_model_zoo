# mtcnn-p

## Use Case and High-Level Description

The `mtcnn-p` model is one of the [mtcnn](https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf) group of models designed to perform face detection. Short for "Multi-task Cascaded Convolutional Neural Network", it is implemented using the Caffe\* framework. The "p" designation indicates that this model is the "proposal" network intended to find the initial set of faces. For details about this family of models, check out the [repository](https://github.com/DuinoDu/mtcnn).

The model input is an image containing the data to be analyzed. The mean values need to be subtracted as follows: [127.5, 127.5, 127.5] before passing the image blob into the network. In addition, values must be divided by 0.0078125.

The model output is a blob with a vector containing the first pass of face data. If there are no faces detected, no further processing is needed. Otherwise, you will typically use this output as input to the `mtcnn-r` model.

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Detection     |
| GFLOPs            | 3.366         |
| MParams           | 0.007         |
| Source framework  | Caffe\*       |

## Accuracy

## Performance

## Input

### Original model

Image, shape - `1,3,720,1280`, format is `B,C,W,H`, where:

- `B` - batch size
- `C` - channel
- `W` - width
- `H` - height

Expected color order: `RGB`.
Mean values - [127.5, 127.5, 127.5], scale value - 128

### Converted model

Image, shape - `1,3,720,1280`, format is `B,C,W,H`, where:

- `B` - batch size
- `C` - channel
- `W` - width
- `H` - height

Expected color order: `RGB`.

## Output

### Original model

1. Face detection, name - `prob1`, shape - `1,2,W,H`, contains scores across two classes (0 - no face, 1 - face) for each pixel whether it contains face or not.
2. Face location, name - `conv4-2`, contains regions with detected faces.

### Converted model

1. Face detection, name - `prob1`, shape - `1,2,W,H`, contains scores across two classes (0 - no face, 1 - face) for each pixel whether it contains face or not.
2. Face location, name - `conv4-2`, contains regions with detected faces.

## Legal Information

[https://raw.githubusercontent.com/DuinoDu/mtcnn/master/LICENSE]()
