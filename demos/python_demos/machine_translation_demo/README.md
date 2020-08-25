# Machine Translation Python\* Demo

This README describes the Machine Translation demo application that uses a non-autoregressive Transformer model for inference.

## How It Works

Upon the start-up the demo application reads command line parameters and loads a network to Inference Engine.

The program provides an interactive CLI interface that gets a sentence in the source language as an input and returns its translation to the target language.

### Running the Demo

Running the application with the `-h` option yields the following usage message:

```
usage: machine_translation_demo.py [-h] -m MODEL --tokenizer-src TOKENIZER_SRC
                                   --tokenizer-tgt TOKENIZER_TGT
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
  --output-name OUTPUT_NAME
                        Optional. Name of the models output node.
```

To run the demo, you can use Intel's pretrained model. To download pretrained models, use the OpenVINO&trade; [Model Downloader](../../../tools/downloader/README.md) or go to the [Intel&reg; Open Source Technology Center](https://download.01.org/opencv/).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

## See Also

* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)