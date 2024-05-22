# BERT Question Answering Embedding Python\* Demo

This README describes the Question Answering Embedding demo application that uses a Squad-tuned BERT model to calculate embedding vectors for context and question to find right context for question. The primary difference from the [bert_question_answering_demo](../../bert_question_answering_demo/python/README.md) is that this demo demonstrates how the inference can be accelerated via pre-computing the embeddings for the contexts.

## How It Works

On startup the demo application reads command line parameters and loads model(s) to OpenVINO™ Runtime plugin.
It also fetches data from the user-provided urls to populate the list of "contexts" with the text.
Prior to the actual inference to answer user's questions, the embedding vectors are pre-calculated (via inference) for each context from the list.
This is done using the first ("embeddings-only") BERT model.

After that, when user types a question, the "embeddings" network is used to calculate an embedding vector for the specified question.
Using the L2 distance between the embedding vector of the question and the embedding vectors for the contexts the best (closest) contexts are selected
as candidates to further seek for the final answer to the question. At this point, the contexts are displayed to the user.

Notice that question is usually much shorter than the contexts, so calculating the embedding for that is really fast. Also calculating the L2 distance between a context and question is almost free, compared to the actual inference. Together, during question answering, this substantially saves on the actual inference, which is needed ONLY for the question (while contexts are pre-calculated), compared to the conventional approach that has to concatenate each context with the question and do an inference on this large input (per context).

If second (conventional SQuAD-tuned) Bert model is provided as well, it is used to further search for the exact answer in the best contexts found in the first step, and the result then also displayed to the user.

## Model API

The demo utilizes model wrappers, adapters and pipelines from [Python* Model API](../../common/python/model_zoo/model_api/README.md).

The generalized interface of wrappers with its unified results representation provides the support of multiple different question answering model topologies in one demo.

## Preparing to Run

The list of models supported by the demo is in `<omz_dir>/demos/bert_question_answering_embedding_demo/python/models.lst` file.
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

* bert-large-uncased-whole-word-masking-squad-0001
* bert-large-uncased-whole-word-masking-squad-emb-0001
* bert-large-uncased-whole-word-masking-squad-int8-0001
* bert-small-uncased-whole-word-masking-squad-0001
* bert-small-uncased-whole-word-masking-squad-0002
* bert-small-uncased-whole-word-masking-squad-emb-int8-0001
* bert-small-uncased-whole-word-masking-squad-int8-0002

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.

## Running

Running the application with the `-h` option yields the following usage message:

```
usage: bert_question_answering_embedding_demo.py [-h] -i INPUT
                                                 [--questions QUESTION [QUESTION ...]]
                                                 [--best_n BEST_N] -v VOCAB
                                                 -m_emb MODEL_EMB
                                                 [--input_names_emb INPUT_NAMES_EMB]
                                                 [-m_qa MODEL_QA]
                                                 [--input_names_qa INPUT_NAMES_QA]
                                                 [--output_names_qa OUTPUT_NAMES_QA]
                                                 [-a MAX_ANSWER_TOKEN_NUM]
                                                 [-d DEVICE] [-c]

Options:
  -h, --help            Show this help message and exit.
  -i INPUT, --input INPUT
                        Required. Urls to a wiki pages with context
  --questions QUESTION [QUESTION ...]
                        Optional. Prepared questions
  --best_n BEST_N       Optional. Number of best (closest) contexts selected
  -v VOCAB, --vocab VOCAB
                        Required. Path to vocabulary file with tokens
  -m_emb MODEL_EMB, --model_emb MODEL_EMB
                        Required. Path to an .xml file with a trained model to
                        build embeddings
  --input_names_emb INPUT_NAMES_EMB
                        Optional. Names for inputs in MODEL_EMB network. For
                        example 'input_ids,attention_mask,token_type_ids','pos
                        ition_ids'
  -m_qa MODEL_QA, --model_qa MODEL_QA
                        Optional. Path to an .xml file with a trained model to
                        give exact answer
  --input_names_qa INPUT_NAMES_QA
                        Optional. Names for inputs in MODEL_QA network. For
                        example 'input_ids,attention_mask,token_type_ids','pos
                        ition_ids'
  --output_names_qa OUTPUT_NAMES_QA
                        Optional. Names for outputs in MODEL_QA network. For
                        example 'output_s,output_e'
  --model_squad_ver MODEL_SQUAD_VER
                        Optional. SQUAD version used for model fine tuning
  -a MAX_ANSWER_TOKEN_NUM, --max_answer_token_num MAX_ANSWER_TOKEN_NUM
                        Optional. Maximum number of tokens in exact answer
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU
                        is acceptable. The demo will look for a suitable plugin
                        for device specified. Default value is CPU
  -c, --colors          Optional. Nice coloring of the questions/answers.
                        Might not work on some terminals (like Windows* cmd
                        console)
  -nireq NUM_INFER_REQUESTS, --num_infer_requests NUM_INFER_REQUESTS
                        Optional. Number of infer requests.
  -nstreams NUM_STREAMS, --num_streams NUM_STREAMS
                        Optional. Number of streams to use for inference on
                        the CPU or/and GPU in throughput mode (for HETERO and
                        MULTI device cases use format
                        <device1>:<nstreams1>,<device2>:<nstreams2> or just
                        <nstreams>).
  -nthreads NUM_THREADS, --num_threads NUM_THREADS
                        Optional. Number of threads to use for inference on
                        CPU (including HETERO cases).
```

## Example Demo Cmd-Line
You can use the following command to try the demo:

```sh
    python3 bert_question_answering_embedding_demo.py
            --vocab=<models_dir>/models/intel/bert-small-uncased-whole-word-masking-squad-0002/vocab.txt
            --model_emb=<path_to_model>/bert-large-uncased-whole-word-masking-squad-emb-0001.xml
            --input_names_emb="input_ids,attention_mask,token_type_ids,position_ids"
            --model_qa=<path_to_model>/bert-small-uncased-whole-word-masking-squad-0002.xml
            --input_names_qa="input_ids,attention_mask,token_type_ids,position_ids"
            --output_names_qa="output_s,output_e"
            --input="https://en.wikipedia.org/wiki/Bert_(Sesame_Street)"
            --input="https://en.wikipedia.org/wiki/Speed_of_light"
            -c
```

The demo will use the Wikipedia articles about the Bert character and the speed of light to answer your questions like
"what is the speed of light", "how to measure the speed of light", "who is Bert", "how old is Bert", etc.

## Demo Inputs

The application reads text from the HTML pages at the given urls and then answers questions typed from the console.
The models and its parameters (inputs and outputs) are also important demo arguments.
Notice that since order of inputs for the model does matter, the demo script checks that the inputs specified
from the command line match the actual network inputs.
The embedding model is reshaped by the demo to infer embedding vectors for long contexts and short question.
Make sure that the original model converted by Model Optimizer with reshape option.
Please see general [reshape intro and limitations](https://docs.openvino.ai/2023.0/_docs_IE_DG_ShapeInference.html)

## Demo Outputs

The application outputs contexts with answers to the same console.
The application reports

* **Latency (all stages)**: total processing time required to process input data (from loading the vocab and processing the context as tokens to displaying the results).
* **Context embeddings latency (stage 1)**: total processing time required to calculate all context embeddings.

## Classifying Documents with Long Texts

Notice that when the original "context" (paragraph text from the url) alone or together with the question do not fit the model input
(usually 384 tokens for the Bert-Large, or 128 for the Bert-Base), the demo splits the paragraph into overlapping segments.
Thus, for the long paragraph texts, the network is called multiple times as for separate contexts.

## See Also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/model_tools/README.md)
* [Benchmark C++ Sample](https://docs.openvino.ai/2023.0/_inference_engine_samples_benchmark_app_README.html)
