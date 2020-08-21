# BERT Question Answering Embedding Python* Demo

This README describes the Question Answering Embedding demo application that uses a Squad-tuned BERT model to calculate embedding vectors for context and question to find right context for question.

## How It Works

Upon the start-up the demo application reads command line parameters and loads a network to Inference engine.
It also fetch data from the user-provided urls to populate the "context" text list.
Then embedding vectors are calculated for each text from list using BERT base model.
Afte that user type the question and the same network is used to calculate embedding vector for question.
Using the distance between question embedding vector and context embedding vectors the best (closest) contexts are choosen
as candidate to have answer for the question.
If second squad tuned bert model for exact answer search is provided then
it is used to search exact answer in the found best contexts


## Running

Running the application with the `-h` option yields the following usage message:
```
python3 question_answering_embedding_demo.py -h
```
The command yields the following usage message:
```
usage: question_answering_embedding_demo.py [-h] -i INPUT [--par_num PAR_NUM]
                                            -v VOCAB --model_emb MODEL_EMB
                                            [--model_qa MODEL_QA]
                                            [--input_names INPUT_NAMES]
                                            [--output_names OUTPUT_NAMES]
                                            [-a MAX_ANSWER_TOKEN_NUM]
                                            [-d DEVICE] [-c]

Options:
  -h, --help            Show this help message and exit.
  -i INPUT, --input INPUT
                        Required. Urls to a wiki pages with context
  --par_num PAR_NUM     Optional. Number of contexts filtered in by embedding
                        vectors
  -v VOCAB, --vocab VOCAB
                        Required. Path to vocabulary file with tokens
  --model_emb MODEL_EMB
                        Required. Path to an .xml file with a trained model to
                        build embeddings
  --model_qa MODEL_QA   Optional. Path to an .xml file with a trained model to
                        give exact answer
  --input_names INPUT_NAMES
                        Optional. Names for inputs in MODEL_QA network. For
                        example 'input_ids,attention_mask,token_type_ids','pos
                        ition_ids'
  --output_names OUTPUT_NAMES
                        Optional. Names for outputs in MODEL_QA network. For
                        example 'output_s,output_e'
  -a MAX_ANSWER_TOKEN_NUM, --max_answer_token_num MAX_ANSWER_TOKEN_NUM
                        Optional. Maximum number of tokens in exact answer
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU
                        is acceptable. Sample will look for a suitable plugin
                        for device specified. Default value is CPU
  -c, --colors          Optional. Nice coloring of the questions/answers.
                        Might not work on some terminals (like Windows* cmd
                        console)

```

> **NOTE**: Before running the demo with a trained model, make sure to convert the model to the Inference Engine's
> Intermediate Representation format (\*.xml + \*.bin)
> using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).
> When using the pre-trained BERT from the model zoo (please see [Model Downloader](../../../tools/downloader/README.md)),
> the model is already converted to the IR.

## Demo Inputs

The application reads text from the HTML pages at the given urls and then answers questions typed from the console.
The models and its parameters (inputs and outputs) are also important demo arguments.
Notice that since order of inputs for the model does matter, the demo script checks that the inputs specified
from the command-line match the actual network inputs.
The embedding model is reshaped by the demo to infer embedding vectors for long contexts and short question.
Be sure that the original model converted by Model Optimizer with reshape option.
Please see general [reshape intro and limitations](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_ShapeInference.html)

## Demo Outputs
The application outputs found answers to the same console.

## Supported Models
[Open Model Zoo Models](../../../models/intel/index.md) feature
example BERT-large tuned on the Squad* for embedding calculation. It comes with "embedding" in its names.
For second stage to find exact answer in filtered context the same models as for question_answering_demo can be used.

## Example Demo Cmd-Line
You can use the following command to try the demo (assuming the model from the Open Model Zoo, downloaded with the
[Model Downloader](../../../tools/downloader/README.md) executed with "--name bert*"):
```
    python3 bert_question_answering_embedding_demo.py
            --vocab=<path_to_model>/vocab.txt
            --model_emb=<path_to_model>/bert-large-uncased-whole-word-masking-squad-embedding-0001.xml
            --model_qa=<path_to_model>/bert-small-uncased-whole-word-masking-squad-0002.xml
            --input_names="input_ids,attention_mask,token_type_ids,position_ids"
            --output_names="output_s,output_e"
            --input="https://en.wikipedia.org/wiki/Bert_(Sesame_Street),https://en.wikipedia.org/wiki/Speed_of_light"
            -c
```
The demo will use a wiki-pages about the Bert character and the speed of light to answer your questions like
"what is the speed of light","how to measure the speed of light" ,"who is Bert", "how old is Bert", etc.

## Classifying Documents with Long Texts
Notice that when the original "context" (paragraph text from the url) alon or together with the question do not fit the model input
(usually 384 tokens for the Bert-Large, or 128 for the Bert-Base), the demo splits the paragraph into overlapping segments.
Thus, for the long paragraph texts, the network is called multiple times as for separate contexts.

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
