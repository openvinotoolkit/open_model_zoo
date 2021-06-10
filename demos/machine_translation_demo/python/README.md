# Machine Translation Python\* Demo

This README describes the Machine Translation demo application that uses a non-autoregressive Transformer model for inference.

## How It Works

On startup the demo application reads command line parameters and loads a network to Inference Engine.

The program provides an interactive CLI interface that gets a sentence in the source language as an input and returns its translation to the target language.

## Preparing to Run

The list of models supported by the demo is in `<omz_dir>/demos/machine_translation_demo/python/models.lst` file.
This file can be used as a parameter for [Model Downloader](../../../tools/downloader/README.md) and Converter to download and, if necessary, convert models to OpenVINO Inference Engine format (\*.xml + \*.bin).

An example of using the Model Downloader:

```sh
python3 <omz_dir>/tools/downloader/downloader.py --list models.lst
```

An example of using the Model Converter:

```sh
python3 <omz_dir>/tools/downloader/converter.py --list models.lst
```

### Supported Models

* machine-translation-nar-de-en-0002
* machine-translation-nar-en-de-0002
* machine-translation-nar-en-ru-0001
* machine-translation-nar-ru-en-0001

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.

## Running the Demo

Running the application with the `-h` option yields the following usage message:

```
usage: machine_translation_demo.py [-h] -m MODEL --tokenizer-src TOKENIZER_SRC
                                   --tokenizer-tgt TOKENIZER_TGT
                                   [-i [INPUT [INPUT ...]]] [-d DEVICE]
                                   [-o OUTPUT] [--output-name OUTPUT_NAME]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model
  --tokenizer-src TOKENIZER_SRC
                        Required. Path to the folder with src tokenizer that
                        contains vocab.json and merges.txt.
  --tokenizer-tgt TOKENIZER_TGT
                        Required. Path to the folder with tgt tokenizer that
                        contains vocab.json and merges.txt.
  -i [INPUT [INPUT ...]], --input [INPUT [INPUT ...]]
                        Optional. Text for translation or path to the input
                        .txt file. Replaces console input.
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU,
                        GPU, HDDL or MYRIAD is acceptable. The demo
                        will look for a suitable plugin for device specified.
                        Default value is CPU.
   -o OUTPUT, --output OUTPUT
                        Optional. Path to the output .txt file.
  --output-name OUTPUT_NAME
                        Optional. Name of the models output node.
```

Required source and target tokenizer files located under `tokenizer_src` and `tokenizer_tgt` subfolders of corresponding model description folder.
For example, to do inference on a CPU with the OpenVINO&trade; toolkit pre-trained models, run the following command:

```sh
python3 ./machine_translation_demo \
  -d CPU \
  -i "a sample english text" \
  -m <path_to_model>/machine-translation-nar-en-de-0002.xml \
  --tokenizer-src <omz_root>/models/intel/machine-translation-nar-en-de-0002/tokenizer_src \
  --tokenizer-tgt <omz_root>/models/intel/machine-translation-nar-en-de-0002/tokenizer_tgt
```

## Demo Output

The application outputs translated sentences from source to target language.

## See Also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
