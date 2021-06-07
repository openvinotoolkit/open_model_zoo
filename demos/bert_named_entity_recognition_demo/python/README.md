# BERT Named Entity Recognition Python\* Demo

This README describes the Named Entity Recognition (NER) demo application that uses a CONLL2003-tuned BERT model for inference.

## How It Works

On startup the demo application reads command line parameters and loads a network to Inference engine.
It also fetch data from the user-provided url to populate the "context" text.
The text is then used to search named entities.

## Preparing to Run

The list of models supported by the demo is in `<omz_dir>/demos/bert_named_entity_recognition_demo/python/models.lst` file.
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

* bert-base-ner

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.

## Running

Running the application with the `-h` option yields the following usage message:

```
usage: bert_named_entity_recognition_demo.py [-h] -v VOCAB -m MODEL -i INPUT
                                             [--input_names INPUT_NAMES]
                                             [-d DEVICE]

Options:
  -h, --help            Show this help message and exit.
  -v VOCAB, --vocab VOCAB
                        Required. path to the vocabulary file with tokens
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model
  -i INPUT, --input INPUT
                        Required. URL to a page with context
  --input_names INPUT_NAMES
                        Optional. Inputs names for the network. Default values
                        are "input_ids,attention_mask,token_type_ids"
  -d DEVICE, --device DEVICE
                        Optional. Target device to perform inference
                        on. Default value is CPU
```

## Demo Inputs

The application reads text from the HTML page at the given URL.
The model and its parameters (inputs and outputs) are also important demo arguments.
Notice that since order of inputs for the model does matter, the demo application checks that the inputs specified
from the command-line match the actual network inputs.

## Demo Outputs

The application outputs recognized named entities (`LOC` - location, `PER` - person, `ORG` - organization, `MISC` - miscellaneous)
for each sentence in input text.

## Example Demo Cmd-Line

You can use the following command to try the demo (assuming the model from the Open Model Zoo, downloaded and converted with the
[Model Downloader](../../../tools/downloader/README.md) executed with "--name bert*"):

```sh
    python3 bert_named_entity_recognition_demo.py.py
            --vocab=<models_dir>/models/public/bert-base-ner/vocab.txt
            --model=<path_to_model>/bert-base-ner.xml
            --input_names="input_ids,attention_mask,token_type_ids"
            --input="https://en.wikipedia.org/wiki/Bert_(Sesame_Street)"
```

## Classifying Documents with Long Texts

Notice that when the original "context" (text from the url) do not fit the model input
(128 for the Bert-Base), the demo reshapes model to maximum sentence length in the "context".

## Demo Performance

Even though the demo reports inference performance (by measuring wall-clock time for individual inference calls),
it is only baseline performance, as certain tricks like batching,
[throughput mode](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Intro_to_Performance.html) can be applied.
Please use the full-blown [Benchmark C++ Sample](https://docs.openvinotoolkit.org/latest/_inference_engine_samples_benchmark_app_README.html)
for any actual performance measurements.

## See Also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
* [Benchmark C++ Sample](https://docs.openvinotoolkit.org/latest/_inference_engine_samples_benchmark_app_README.html)
