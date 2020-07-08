# BERT Question Answering Python* Demo

This README describes the Question Answering demo application that uses a Squad-tuned BERT model for inference.

## How It Works

Upon the start-up the demo application reads command line parameters and loads a network to Inference engine.
It also fetch data from the user-provided url to populate the "context" text.
The text is then used to search answers for user-provided questions.


## Running

Running the application with the `-h` option yields the following usage message:
```
python3 question_answering_demo.py -h
```
The command yields the following usage message:
```
usage: question_answering_demo.py [-h] -v VOCAB -m MODEL -i INPUT
                                  [--input_names INPUT_NAMES]
                                  [--output_names OUTPUT_NAMES]
                                  [--model_squad_ver MODEL_SQUAD_VER]
                                  [-a MAX_ANSWER_TOKEN_NUM] [-d DEVICE]

Options:
  -h, --help            Show this help message and exit.
  -v VOCAB, --vocab VOCAB
                        Required. path to vocabulary file with tokens
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model
  -i INPUT, --input INPUT
                        Required. Url to a page with context
  --input_names INPUT_NAMES
                        Optional. Inputs names for the  network.
                        Default values are "input_ids,attention_mask,token_type_ids"
  --output_names OUTPUT_NAMES
                        Required. Outputs names for the network.
                        Default values are "output_s,output_e"
  --model_squad_ver MODEL_SQUAD_VER
                        Optional. SQUAD version used for model fine tuning
  -q MAX_QUESTION_TOKEN_NUM, --max_question_token_num MAX_QUESTION_TOKEN_NUM
                        Optional. Maximum number of tokens in question (used with the reshape option)
  -a MAX_ANSWER_TOKEN_NUM, --max_answer_token_num MAX_ANSWER_TOKEN_NUM
                        Optional. Maximum number of tokens in answer
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU
                        is acceptable. Sample will look for a suitable plugin
                        for device specified. Default value is CPU
  -r, --reshape
                        Optional. Auto reshape sequence length
                                  to the input context + max question len (to improve the speed)
  -c, --colors
                        Optional. Nice coloring of the questions/answers.
                        Might not work on some terminals (like Windows* cmd console)

```

> **NOTE**: Before running the demo with a trained model, make sure to convert the model to the Inference Engine's
> Intermediate Representation format (\*.xml + \*.bin)
> using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).
> When using the pre-trained BERT from the model zoo (please see [Model Downloader](../../../tools/downloader/README.md)),
> the model is already converted to the IR.

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

## Supported Models
[Open Model Zoo Models](../../../models/intel/index.md) feature
example BERT-large trained on the Squad*.
One specific flavor of that is so called "distilled" model (for that reason it comes with "small" in its name,
but don't get confused as it is still originated from the BERT Large) that is indeed substantially smaller and faster.

The demo also works fine with [official MLPerf* BERT ONNX models fine-tuned on the Squad dataset](
https://github.com/mlperf/inference/tree/master/v0.7/language/bert).
Unlike [[Open Model Zoo Models](../../../models/intel/index.md) that come directly as the
Intermediate Representation (IR), the MLPerf models should be explicitly converted with
[OpenVINO Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).
Specifically the example command-line (for the int8 model) is as follows:
```
    python3 mo.py
            -m <path_to_model>/bert_large_v1_1_fake_quant.onnx
            --input "input_ids,attention_mask,token_type_ids"
            --input_shape "[1,384],[1,384],[1,384]"
            --keep_shape_ops
```

## Example Demo Cmd-Line
You can use the following command to try the demo (assuming the model from the Open Model Zoo, downloaded with the
[Model Downloader](../../../tools/downloader/README.md) executed with "--name bert*"):
```
    python3 bert_question_answering_demo.py
            --vocab=<path_to_model>/vocab.txt
            --model=<path_to_model>/bert-small-uncased-whole-word-masking-squad-0001.xml
            --input_names="input_ids,attention_mask,token_type_ids"
            --output_names="output_s,output_e"
            --input="https://en.wikipedia.org/wiki/Bert_(Sesame_Street)"
            -c
```
The demo will use a wiki-page about the Bert character to answer your questions like "who is Bert", "how old is Bert", etc.

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
* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
* [Benchmark C++ Sample](https://docs.openvinotoolkit.org/latest/_inference_engine_samples_benchmark_app_README.html)
