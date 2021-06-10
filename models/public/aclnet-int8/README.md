# aclnet-int8

## Use Case and High-Level Description

The `AclNet-int8` model is quantized and fine-tuned with NNCF variant of [AclNet](../aclnet/README.md) model, which is designed to perform sound classification.
The `AclNet-int8` model is trained on an internal dataset of environmental sounds for 53 different classes, listed in file `<omz_dir>/data/dataset_classes/aclnet_53cl.txt`.
For details about the model, see this [paper](https://arxiv.org/abs/1811.06669).

The model input is a segment of PCM audio samples in `N, C, 1, L` format.

The model output for `AclNet-int8` is the sound classifier output for the 53 different environmental sound classes from the internal sound database.

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 2.71          |
| MParams           | 1.41          |
| Source framework  | PyTorch\*     |

## Accuracy

| Metric | Value |
| ------ | ----- |
| Top 1  | 87.1% |
| Top 5  | 93.0% |

Metrics were computed on internal validation dataset according to following [publication](http://dcase.community/documents/workshop2019/proceedings/DCASE2019Workshop_Huang_52.pdf) and [paper](https://arxiv.org/abs/1811.06669).

## Input

### Original Model

Audio, name - `result.1`, shape - `1, 1, 1, L`, format is `N, C, 1, L`, where:

- `N` - batch size
- `C` - channel
- `L` - number of PCM samples (minimum value is 16000)

### Converted Model

Audio, name - `result.1`, shape - `1, 1, 1, L`, format is `N, C, 1, L`, where:

- `N` - batch size
- `C` - channel
- `L` - number of PCM samples (minimum value is 16000)

## Output

### Original Model

Sound classifier (see labels file, `<omz_dir>/data/dataset_classes/aclnet_53cl.txt`), name - `486`, shape - `1, 53`, output data format is `N, C`, where:

- `N` - batch size
- `C` - predicted softmax scores for each class in [0, 1] range

### Converted Model

Sound classifier (see labels file, `<omz_dir>/data/dataset_classes/aclnet_53cl.txt`), name - `486`, shape - `1, 53`, output data format is `N, C`, where:

- `N` - batch size
- `C` - predicted softmax scores for each class in [0, 1] range

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

The original model is distributed under [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0.html).
