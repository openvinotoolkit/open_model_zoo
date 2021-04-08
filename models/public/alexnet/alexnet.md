# alexnet

## Use Case and High-Level Description

The `alexnet` model is designed to perform image classification. Just like other common classification models, the `alexnet` model has been pre-trained on the ImageNet image database. For details about this model, check out the [paper](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf).

The model input is a blob that consists of a single image of `1, 3, 227, 227` in `BGR` order. The BGR mean values need to be subtracted as follows: [104, 117, 123] before passing the image blob into the network.

The model output for `alexnet` is the usual object classifier output for the 1000 different classifications matching those in the ImageNet database.

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 1.5           |
| MParams           | 60.965        |
| Source framework  | Caffe\*       |

## Accuracy

| Metric | Value   |
| ------ | ------- |
| Top 1  | 56.598% |
| Top 5  | 79.812% |

See [the original model's documentation](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet).

## Input

### Original model

Image, name - `data`, shape - `1, 3, 227, 227`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.
Mean values - [104, 117, 123]

### Converted model

Image, name - `data`, shape - `1, 3, 227, 227`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original model

Object classifier according to ImageNet classes, name - `prob`, shape - `1, 1000`, output data format is `B, C`, where:

- `B` - batch size
- `C` - predicted probabilities for each class in  [0, 1] range

### Converted model

Object classifier according to ImageNet classes, name - `prob`, shape - `1, 1000`, output data format is `B, C`, where:

- `B` - batch size
- `C` - predicted probabilities for each class in  [0, 1] range


## Download a Model and Convert it into Inference Engine Format

You can download models and if necessary convert them into Inference Engine format using the [Model Downloader and other automation tools](../../../tools/downloader/README.md) as shown in the examples below.

An example of using the Model Downloader:
```
python3 <omz_dir>/tools/downloader/downloader.py --name <model_name>
```

An example of using the Model Converter:
```
python3 <omz_dir>/tools/downloader/converter.py --name <model_name>
```

## Legal Information

The original model is distributed under the following
[license](https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/readme.md):

```
This model is released for unrestricted use.
```
