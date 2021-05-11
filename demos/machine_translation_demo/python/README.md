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
                        GPU, FPGA, HDDL or MYRIAD is acceptable. The demo
                        will look for a suitable plugin for device specified.
                        Default value is CPU.
   -o OUTPUT, --output OUTPUT
                        Optional. Path to the output .txt file.
  --output-name OUTPUT_NAME
                        Optional. Name of the models output node.
```

To run the demo, you can use Intel's pretrained model. To download pretrained models, use the OpenVINO&trade; [Model Downloader](../../../tools/downloader/README.md). The list of models supported by the demo is in `<omz_dir>/demos/machine_translation_demo/python/models.lst`.

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

## See Also

* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
