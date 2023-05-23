# GPT-2 Text Prediction Python\* Demo

This demo shows how to use gpt-2 model for inference to perform interactive conditional text prediction, where content is generated based on text provided by user.

## How It Works

On startup the demo application reads command line parameters and loads a model to OpenVINO™ Runtime plugin.
It also encodes a user input prompt received via command line arguments or user input, and then uses it to predict the output sequence.

## Preparing to Run

The list of models supported by the demo is in `<omz_dir>/demos/gpt2_text_prediction_demo/python/models.lst` file.
This file can be used as a parameter for [Model Downloader](../../../tools/model_tools/README.md) and Converter to download and, if necessary, convert models to OpenVINO IR format (\*.xml + \*.bin).

An example of using the Model Downloader:

```sh
omz_downloader --list models.lst
```

An example of using the Model Converter:

```sh
omz_converter --list models.lst
```

### Supported Models

* gpt-2

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.

## Running

Running the application with the `-h` option yields the following usage message:

```
usage: gpt2_text_prediction_demo.py [-h] -m MODEL -v VOCAB --merges MERGES
                                    [-i INPUT]
                                    [--max_sample_token_num MAX_SAMPLE_TOKEN_NUM]
                                    [--top_k TOP_K] [--top_p TOP_P]
                                    [-d DEVICE]

Options:
  -h, --help            Show this help message and exit.
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model
  -v VOCAB, --vocab VOCAB
                        Required. Path to the vocabulary file with tokens
  --merges MERGES       Required. Path to the merges file
  -i INPUT, --input INPUT
                        Optional. Input prompt
  --max_sample_token_num MAX_SAMPLE_TOKEN_NUM
                        Optional. Maximum number of tokens in generated sample
  --top_k TOP_K         Optional. Number of tokens with the highest
                        probability which will be kept for generation
  --top_p TOP_P         Optional. Maximum probability, tokens with such a
                        probability and lower will be kept for generation
  -d DEVICE, --device DEVICE
                        Optional. Target device to perform inference
                        on. Default value is CPU
  --dynamic_shape       Run model with dynamic input sequence. If not
                        provided, input sequence will be padded to
                        max_seq_len
  --max_seq_len MAX_SEQ_LEN
                        Optional. Maximum sequence length for processing. Default value is 1024
```

## Demo Inputs

The application reads and encodes text from input string, then performs transformations and uses it as model input.

## Demo Outputs

The application outputs predicted text, continuing input string for each input string.

## Example Demo Cmd-Line

You can use the following command to try the demo (assuming the used model from the Open Model Zoo, downloaded and converted with the
[Model Downloader](../../../tools/model_tools/README.md)):

```sh
    python3 gpt2_text_prediction_demo.py
            --model=<path_to_model>/gpt-2.xml
            --vocab=<models_dir>/models/public/gpt-2/gpt2/vocab.json
            --merges=<models_dir>/models/public/gpt-2/gpt2/merges.txt
```

## See Also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/model_tools/README.md)
