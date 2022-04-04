# vitstr-small-patch16-224

## Use-case and high-level description

The `vitstr-small-patch16-224` model is `small` version of the ViTSTR models. ViTSTR is a simple single-stage model that uses a pre-trained Vision Transformer (ViT) to perform Scene Text Recognition (ViTSTR). Small version of model has an embedding size of 384 and number of heads of 6. Model is able to recognize alphanumeric case sensitive text and special characters.

More details provided in the [paper](https://arxiv.org/abs/2105.08582) and [repository](https://github.com/roatienza/deep-text-recognition-benchmark).

## Specification

| Metric           | Value                  |
| ---------------- | ---------------------- |
| Type             | Scene Text Recognition |
| GFLOPs           | 9.1544                 |
| MParams          | 21.5061                |
| Source framework | PyTorch\*              |

## Accuracy

Alphanumeric subset of common scene text recognition benchmarks are used. For your convenience you can see dataset size. Note, that we use here ICDAR15 alphanumeric subset without irregular (arbitrary oriented, perspective or curved) texts. See details [here](https://arxiv.org/abs/1709.02054), section 4.1. All reported results are achieved without using any lexicon.

| Dataset  | Accuracy | Dataset size |
| -------- | -------- | ------------ |
| ICDAR-03 | 93.43%   | 867          |
| ICDAR-13 | 90.34%   | 1015         |
| ICDAR-15 | 75.04%   | 1811         |
| SVT      | 85.47%   | 647          |
| IIIT5K   | 87.07%   | 3000         |

Use `accuracy_check [...] --model_attributes <path_to_folder_with_downloaded_model>` to specify the path to additional model attributes. `path_to_folder_with_downloaded_model` is a path to the folder, where the current model is downloaded by [Model Downloader](../../../tools/model_tools/README.md) tool.

## Input

### Original model

Image, name: `image`, shape: `1, 1, 224, 224` in the format `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Note that the source image should be tight aligned crop with detected text converted to grayscale.

Scale values - [255].

### Converted model

Image, name: `image`, shape: `1, 1, 224, 224` in the format `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Note that the source image should be tight aligned crop with detected text converted to grayscale.

## Output

### Original model

Output tensor, name: `logits`, shape: `1, 25, 96` in the format `B, W, L`, where:

- `B` - batch size
- `W` - output sequence length
- `L` - confidence distribution across [GO] - special start token for decoder, [s] - special end of sequence character for decoder and characters, listed in enclosed file `vocab.txt`.

The network output decoding process is pretty easy: get the argmax on `L` dimension, transform indices to letters and slice the resulting phrase on the first entry of `end-of-sequence` symbol.

### Converted model

Output tensor, name: `logits`, shape: `1, 25, 96` in the format `B, W, L`, where:

- `B` - batch size
- `W` - output sequence length
- `L` - confidence distribution across [GO] - special start token for decoder, [s] - special end of sequence character for decoder and characters, listed in enclosed file `vocab.txt`.

The network output decoding process is pretty easy: get the argmax on `L` dimension, transform indices to letters and slice the resulting phrase on the first entry of `end-of-sequence` symbol.

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

* [Text Detection C++ Demo](../../../demos/text_detection_demo/cpp/README.md)

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/roatienza/deep-text-recognition-benchmark/master/LICENSE.md).
A copy of the license is provided in `<omz_dir>/models/public/licenses/APACHE-2.0.txt`.
