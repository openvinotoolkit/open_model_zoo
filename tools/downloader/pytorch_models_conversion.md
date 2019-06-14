# Conversion pretrained models from PyTorch to ONNX

Models from PyTorch framework not supported by Model Optimizer now. Before conversion to IR, they
must be converted to ONNX using script `pytorch_to_innx.py`. The script enables conversion of models 
from torchvision and for public models, which description and pre-trained weights are
available for cloning or downloading from the internet resources.

## Supported models

The models, supported in the current version of the script, are:

* Torchvision models:
    * ResNet-50-v1
    * Inception-v3

* Public pre-trained models
    * MobileNetV2 (<https://github.com/tonylins/pytorch-mobilenet-v2>)

## Setup

### Prerequisites

The script was developed and tested with

* Ubuntu 16.04
* Python 3.5.2
* Torch 1.0.1
* Torchvision 0.2.2 (for torchvision models support)
* ONNX 1.4.1

### Installation

> **Note:** It is recommended to use this script under python virtual environment to avoid possible conflicts between
> already installed python packages and required packages for script.

To use virtual environment, create and activate it:

```bash
python3 -m virtualenv -p `which python3` <directory_for_environment>
source <directory_for_environment>/bin/activate
```
Install requirements:
```bash
pip3 install torch==1.0.1 torchvision>=0.2.2 onnx==1.4.1 numpy<1.15
```
Now you can work with the script.

To deactivate virtual environment after finishing work, use the following command:

```bash
deactivate
```

## Usage

The script takes the following input arguments:

* `--model-name` - name of the PyTorch model to convert. Currently available model names are:
    * resnet-v1-50
    * inception-v3
    * mobilenet-v2
* `--weights` - path to a .pth or .pth.tar file with downloaded pre-trained PyTorch weights
* `--input-shape` - input blob shape, given by comma-separated positive integer values for `batch size`,
  `number of channels`, `height` and `width` in the order, defined for the chosen model
* `--output-file` - path to the output .onnx file with the converted model.

Optional arguments, taken by the script:

* `--model-path` - path to a directory with python file(s), containing description of PyTorch model, chosen for
  conversion. This parameter should be provided for public models, that are not a part of torchvision package.
* `--input-names` - space-separated (if several) names of input layers. The input layers' names would be presented by
  these values in ONNX model, or indexes of layers would be used instead, if this argument was not provided.
* `--output-names` - space-separated (if several) names of output layers. The output layers' names would be presented
  by these values in ONNX model, or indexes of layers would be used instead, if this argument was not provided.

You may also refer to `-h, --help` option for getting full list of script arguments.

For example, to convert ResNet-50-v1 model from torchvision, the following command may be used:

```bash
python3 pytorch_to_onnx.py \
    --model-name resnet-v1-50 \
    --weights  <path_to_downloaded_pretrained_weights>/resnet50-19c8e357.pth \
    --input-shape 1 3 224 224 \
    --output-file <path_to_save_converted_model>/resnet-v1-50.onnx \
    --input-names data \
    --output-names prob
```
