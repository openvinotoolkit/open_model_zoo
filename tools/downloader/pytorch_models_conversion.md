# Conversion pretrained models from PyTorch to ONNX

Models from PyTorch framework not supported by Model Optimizer now. Before conversion to IR, they
must be converted to ONNX using script `pytorch_to_onnx.py`. The script enables conversion of models
from torchvision and for public models, which description and pre-trained weights are
available for cloning or downloading from the internet resources.

## Supported models

Any model may be converted by this script. To do this weights and model's source code is needed.

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
pip3 install torch==1.0.1 onnx==1.4.1 numpy<1.15
```
Additionally install TorchVision\* if necessary:
```bash
pip3 install torchvision>=0.2.2
```

Now you can work with the script.

To deactivate virtual environment after finishing work, use the following command:

```bash
deactivate
```

## Usage

```
Conversion of pretrained models from PyTorch to ONNX

optional arguments:
  -h, --help            show this help message and exit
  --model-name MODEL_NAME
                        Model to convert. May be class name or name of
                        constructor function
  --weights WEIGHTS     Path to the weights in PyTorch's format
  --input-shape INPUT_DIM
                        Shape of the input blob
  --output-file OUTPUT_FILE
                        Path to the output ONNX model
  --from-torchvision    Sets model's origin as Torchvision*
  --model-path MODEL_PATH
                        Path to PyTorch model's source code if model is not
                        from Torchvision*
  --import-module IMPORT_MODULE
                        Name of module, which contains model's
                        constructor.Requires if model not from Torchvision
  --input-names INPUT_NAMES [INPUT_NAMES ...]
                        Space separated names of the input layers
  --output-names OUTPUT_NAMES [OUTPUT_NAMES ...]
                        Space separated names of the output layers
```

For example, to convert ResNet-50-v1 model from torchvision, the following command may be used:

```bash
python3 pytorch_to_onnx.py \
    --model-name resnet50 \
    --from-torchvision \
    --weights  <path_to_downloaded_pretrained_weights>/resnet50-19c8e357.pth \
    --input-shape 1,3,224,224 \
    --output-file <path_to_save_converted_model>/resnet-v1-50.onnx \
    --input-names data \
    --output-names prob
```
