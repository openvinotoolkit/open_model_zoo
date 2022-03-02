# aclnet

## Use Case and High-Level Description

The `AclNet` model is designed to perform sound classification and is trained on internal dataset of environmental sounds for 53 different classes, listed in file `<omz_dir>/data/dataset_classes/aclnet_53cl.txt`.
For details about the model, see this [paper](https://arxiv.org/abs/1811.06669).

The model input is a segment of PCM audio samples in `N, C, 1, L` format.

The model output for `AclNet` is the sound classifier output for the 53 different environmental sound classes from the internal sound database.

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 1.42          |
| MParams           | 2.71          |
| Source framework  | PyTorch\*     |

## Accuracy

| Metric | Value |
| ------ | ----- |
| Top 1  | 86.3% |
| Top 5  | 92.0% |

Metrics were computed on internal validation dataset according to following [publication](http://dcase.community/documents/workshop2019/proceedings/DCASE2019Workshop_Huang_52.pdf) and [paper](https://arxiv.org/abs/1811.06669).

## Input

### Original Model

Audio, name - `input`, shape - `1, 1, 1, L`, format is `N, C, 1, L`, where:

- `N` - batch size
- `C` - channel
- `L` - number of PCM samples (minimum value is 16000)

### Converted Model

Audio, name - `input`, shape - `1, 1, 1, L`, format is `N, C, 1, L`, where:

- `N` - batch size
- `C` - channel
- `L` - number of PCM samples (minimum value is 16000)

## Output

### Original Model

Sound classifier (see labels file, `<omz_dir>/data/dataset_classes/aclnet_53cl.txt`), name - `output`, shape - `1, 53`, output data format is `N, C`, where:

- `N` - batch size
- `C` - predicted softmax scores for each class in [0, 1] range

### Converted Model

Sound classifier (see labels file, `<omz_dir>/data/dataset_classes/aclnet_53cl.txt`), name - `output`, shape - `1, 53`, output data format is `N, C`, where:

- `N` - batch size
- `C` - predicted softmax scores for each class in [0, 1] range

## Download a Model and Convert it into OpenVINO™ IR Format

You can download models and if necessary convert them into OpenVINO™ IR format using the [Model Downloader and other automation tools](../../../tools/model_tools/README.md) as shown in the examples below.

An example of using the Model Downloader:
```
omz_downloader --name <model_name>
```

An example of using the Model Converter:
```
omz_converter --name <model_name>
```

## Demo usage

The model can be used in the following demos provided by the Open Model Zoo to show its capabilities:

* [Sound Classification Python\* Demo](../../../demos/sound_classification_demo/python/README.md)

## Legal Information

The original model is distributed under [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0.html).
