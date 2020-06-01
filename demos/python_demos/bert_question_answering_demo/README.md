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
usage: question_answering_demo.py [-h] -v VOCAB -m MODEL --input_names
                                  INPUT_NAMES --output_names OUTPUT_NAMES
                                  [--model_squad_ver MODEL_SQUAD_VER] -i INPUT
                                  [-a MAX_ANSWER_TOKEN_NUM] [-d DEVICE]

Options:
  -h, --help            Show this help message and exit.
  -v VOCAB, --vocab VOCAB
                        Required. path to vocabulary file with tokens
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model
  --input_names INPUT_NAMES
                        Required. Names for inputs in networks. For example
                        ['input_ids','attention_mask','token_type_ids']
  --output_names OUTPUT_NAMES
                        Required. Names for outputs in networks. For example
                        ['output_s','output_e']
  --model_squad_ver MODEL_SQUAD_VER
                        Optional. SQUAD version used for model fine tuning
  -i INPUT, --input INPUT
                        Required. Url to a page with context
  -q MAX_QUESTION_TOKEN_NUM, --max_question_token_num MAX_QUESTION_TOKEN_NUM
                        Optional. Maximum number of tokens in question (used with the reshape option)
  -r, --reshape
                        Optional. Auto reshape sequence length
                                  to the input context + max question len (to improve the speed)
  -a MAX_ANSWER_TOKEN_NUM, --max_answer_token_num MAX_ANSWER_TOKEN_NUM
                        Optional. Maximum number of tokens in answer
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU
                        is acceptable. Sample will look for a suitable plugin
                        for device specified. Default value is CPU

```

> **NOTE**: Before running the demo with a trained model, make sure to convert the model to the Inference Engine's
> Intermediate Representation format (\*.xml + \*.bin)
> using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).
> When using the pre-trained BERT from the model zoo (please see [Model Downloader](../../../tools/downloader/README.md)),
> the model is already converted to the IR.

## Demo Inputs

The application reads text from the html page by the given url and then answers questions typed from the console.
The model and it's parameters (inputs and outputs) is another important demo arguments.
Notice that since order of inputs for the model does matter, the demo script checks that the inputs specified
from the command-line match the actual network inputs.
When reshape ('-r') reshape option is specified, the script also attempts to reshape the network to the
length of the context plus length of the question (both in tokens), if the resulting value is smaller than the original
sequence length that the network expects. This is performance (speed) and memory footprint saving option.
Since some networks are not-reshapable (due to limitations of the internal layers) the reshaping might fail,
so you will need to run the demo without it.
Please see general ['reshape intro and limitations'](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_ShapeInference.html)

## Demo Outputs

The application outputs found answers to the same console.

## Example Cmd-Line
You can use the following command to try the demo (assuming fp32 model from the Open Model Zoo, downloaded with the
[Model Downloader](../../../tools/downloader/README.md)):
```
    python3 question_answering_demo.py
            --vocab=<path_to_model>/vocab.txt
            --model=<path_to_model>/bert-large-uncased-whole-word-masking-squad-fp32-onnx-0001.xml
            --input_names="['0','1','2']"
            --output_names="['3171','3172']"
            --input="https://en.wikipedia.org/wiki/Bert_(Sesame_Street)"
```
The demo will use a wiki-page about the Bert character to answer your questions like "who is Bert", "how old is Bert", etc.

## Classifying Documents with Long Texts
Notice that when the original "context" (text from the url) together with the question do not fit the model input
(usually 384 tokens for the Bert-Large, or 128 for the Bert-Base), the demo splits the context into overlapping segments.
Thus, for the long texts, the network is called multiple times. The results are then sorted by the probabilities.

## Demo Performance
Even though, the demo reports inference performance (by measuring wall-clock time for individual inference calls),
it is rather baseline performance, as certain tricks like batching,
[throughput mode] (https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Intro_to_Performance.html) can be applied.
Please use the full-blown [Benchmark C++ Sample](https://docs.openvinotoolkit.org/latest/_inference_engine_samples_benchmark_app_README.html)
for any actual performance measurements.


## See Also
* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
* [Benchmark C++ Sample](https://docs.openvinotoolkit.org/latest/_inference_engine_samples_benchmark_app_README.html)
