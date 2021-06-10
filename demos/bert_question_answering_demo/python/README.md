# BERT Question Answering Python\* Demo

This README describes the Question Answering demo application that uses a Squad-tuned BERT model for inference.

## How It Works

On startup the demo application reads command line parameters and loads a network to Inference engine.
It also fetches data from the user-provided url to populate the "context" text.
The text is then used to search answers for user-provided questions.

## Preparing to Run

The list of models supported by the demo is in `<omz_dir>/demos/bert_question_answering_demo/python/models.lst` file.
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

* bert-large-uncased-whole-word-masking-squad-0001
* bert-large-uncased-whole-word-masking-squad-emb-0001
* bert-large-uncased-whole-word-masking-squad-int8-0001
* bert-small-uncased-whole-word-masking-squad-0001
* bert-small-uncased-whole-word-masking-squad-0002
* bert-small-uncased-whole-word-masking-squad-emb-int8-0001
* bert-small-uncased-whole-word-masking-squad-int8-0002

The "small" variants of these are so-called "distilled" models, which originated from the BERT Large but substantially smaller and faster.
If you want to use an official MLPerf* BERT ONNX model rather than the distilled model on the Open model Zoo, the command line to convert the [int8 model](https://zenodo.org/record/3750364) is as follows:

```sh
    python3 mo.py
            -m <path_to_model>/bert_large_v1_1_fake_quant.onnx
            --input "input_ids,attention_mask,token_type_ids"
            --input_shape "[1,384],[1,384],[1,384]"
            --keep_shape_ops
```

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.

## Running

Running the application with the `-h` option yields the following usage message:

```
usage: bert_question_answering_demo.py [-h] -v VOCAB -m MODEL -i INPUT
                                       [--questions QUESTION [QUESTION ...]]
                                       [--input_names INPUT_NAMES]
                                       [--output_names OUTPUT_NAMES]
                                       [--model_squad_ver MODEL_SQUAD_VER]
                                       [-q MAX_QUESTION_TOKEN_NUM]
                                       [-a MAX_ANSWER_TOKEN_NUM] [-d DEVICE]
                                       [-r] [-c]

Options:
  -h, --help            Show this help message and exit.
  -v VOCAB, --vocab VOCAB
                        Required. path to the vocabulary file with tokens
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model
  -i INPUT, --input INPUT
                        Required. URL to a page with context
  --questions QUESTION [QUESTION ...]
                        Optional. Prepared questions
  --input_names INPUT_NAMES
                        Optional. Inputs names for the network. Default values
                        are "input_ids,attention_mask,token_type_ids"
  --output_names OUTPUT_NAMES
                        Optional. Outputs names for the network. Default
                        values are "output_s,output_e"
  --model_squad_ver MODEL_SQUAD_VER
                        Optional. SQUAD version used for model fine tuning
  -q MAX_QUESTION_TOKEN_NUM, --max_question_token_num MAX_QUESTION_TOKEN_NUM
                        Optional. Maximum number of tokens in question
  -a MAX_ANSWER_TOKEN_NUM, --max_answer_token_num MAX_ANSWER_TOKEN_NUM
                        Optional. Maximum number of tokens in answer
  -d DEVICE, --device DEVICE
                        Optional. Target device to perform inference
                        on.Default value is CPU
  -r, --reshape         Optional. Auto reshape sequence length to the input
                        context + max question len (to improve the speed)
  -c, --colors          Optional. Nice coloring of the questions/answers.
                        Might not work on some terminals (like Windows* cmd
                        console)

```

## Example Demo Cmd-Line

```sh
    python3 bert_question_answering_demo.py
            --vocab=<omz_dir>/models/intel/bert-small-uncased-whole-word-masking-squad-0001/vocab.txt
            --model=<path_to_model>/bert-small-uncased-whole-word-masking-squad-0001.xml
            --input_names="input_ids,attention_mask,token_type_ids"
            --output_names="output_s,output_e"
            --input="https://en.wikipedia.org/wiki/Bert_(Sesame_Street)"
            -c
```

The demo will use a wiki-page about the Bert character to answer your questions like "who is Bert", "how old is Bert", etc.

## Demo Inputs

The application reads text from the HTML page at the given url and then answers questions typed from the console.
The model and its parameters (inputs and outputs) are also important demo arguments.
Notice that since order of inputs for the model does matter, the demo script checks that the inputs specified
from the command-line match the actual network inputs.
When the reshape option (`-r`)  is specified, the script also attempts to reshape the network to the
length of the context plus length of the question (both in tokens), if the resulting value is smaller than the original
sequence length that the network expects. This is performance (speed) and memory footprint saving option.
Since some networks are not-reshapable (due to limitations of the internal layers) the reshaping might fail,
so you will need to run the demo without it.
Please see general [reshape intro and limitations](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_ShapeInference.html)

## Demo Outputs

The application outputs found answers to the same console.

## Classifying Documents with Long Texts

Notice that when the original "context" (text from the url) together with the question do not fit the model input
(usually 384 tokens for the Bert-Large, or 128 for the Bert-Base), the demo splits the context into overlapping segments.
Thus, for the long texts, the network is called multiple times. The results are then sorted by the probabilities.

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
