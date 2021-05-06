# Machine Translation Python\* Demo

This README describes the Machine Translation demo application that uses a non-autoregressive Transformer model for inference.

## How It Works

Upon the start-up the demo application reads command line parameters and loads a network to Inference Engine.

The program provides an interactive CLI interface that gets a sentence in the source language as an input and returns its translation to the target language.

## Preparing to run

Pre-trained models, supported by demo listed in [models.lst](./models.lst) file, located at each demo folder.
This file can be used as a parameter for [Model Downloader](../../../tools/downloader/README.md) and Converter to download and, if necessary, convert models to OpenVINO Inference Engine format (\*.xml + \*.bin).

### Supported models

* machine-translation-nar-de-en-0001
* machine-translation-nar-en-de-0001
* machine-translation-nar-en-ru-0001
* machine-translation-nar-ru-en-0001

> **NOTE**: Refer to tables for [Intel](../../../models/intel/device_support.md) and [public](../../../models/public/device_support.md) models which summarize models support at different devices to select target inference device.

### Running the Demo

Running the application with the `-h` option yields the following usage message:

```
usage: machine_translation_demo.py [-h] -m MODEL --tokenizer-src TOKENIZER_SRC
                                   --tokenizer-tgt TOKENIZER_TGT
                                   [-i [INPUT [INPUT ...]]]
                                   [--output-name OUTPUT_NAME]

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
                        Optional. Text for translation. Replaces console input.
  --output-name OUTPUT_NAME
                        Optional. Name of the models output node.
```

Required source and target tokenizer files located under `tokenizer_src` and `tokenizer_tgt` subfolders of corresponding model description folder.
For example, to do inference on a CPU with the OpenVINO&trade; toolkit pre-trained models, run the following command:

```sh
python3 ./machine_translation_demo -i "a sample english text" -m <path_to_model>/machine-translation-nar-en-de-0001.xml --tokenizer-src <omz_root>/models/intel/machine-translation-nar-en-de-0001/tokenizer_src --tokenizer-tgt <omz_root>/models/intel/machine-translation-nar-en-de-0001/tokenizer_tgt
```

## See Also

* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
