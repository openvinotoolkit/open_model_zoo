# BERT Question Answering Embedding Python\* Demo

This README describes the Question Answering Embedding demo application that uses a Squad-tuned BERT model to calculate embedding vectors for context and question to find right context for question. The primary difference from the [bert_question_answering_demo](../bert_question_answering_demo/README.md) is that this demo domonstrates how the inference can be accelerated via pre-computing the embeddings for the contexts.

## How It Works

Upon the start-up the demo application reads command line parameters and loads network(s) to the InferenceEngine.
It also fetches data from the user-provided urls to populate the list of "contexts" with the text.
Prior to the actual inference to answer user's questions, the embedding vectors are pre-calculated (via inference) for each context from the list.
This is done using the first ("emdbeddings-only") BERT model.

After that, when user type the question and the "embeddings" network is used to calculate an embedding vector for the specified question.
Using the L2 distance between the embedding vector of the question and the embedding vectors for the contexts the best (closest) contexts are selected
as candidates to further seek for the final answer to the question. At this point, the contexts are displayed to the user.

Notice that question is usually much shorter than the contexts, so calculating the embedding for that is really fast. Also calculating the L2 distance between a context and question is almost free, compared to the actual inference. Together, during question answering, this substantially saves on the actual inference, which is needed ONLY for the question (while contexts are pre-calculated), compared to the conventional approach that has to concatenate each context with the question and do an inference on this large input (per context).

If second (conventional SQuAD-tuned) Bert model is provided as well, it is used to further search for the exact answer in the best contexts found in the first step, and the result then also displayed to the user.

## Running

Running the application with the `-h` option yields the following usage message:
```
usage: bert_question_answering_embedding_demo.py [-h] -i INPUT
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
The application outputs contexts with answers to the same console.

## Supported Models
[Open Model Zoo Models](../../../models/intel/index.md) feature
example BERT-large tuned on the Squad* for embedding calculation. It comes with "embedding" in its name.
For second stage to find exact answer in filtered context the same models as for `bert_question_answering_demo` can be used.

## Example Demo Cmd-Line
You can use the following command to try the demo (assuming the model from the Open Model Zoo, downloaded with the
[Model Downloader](../../../tools/downloader/README.md) executed with "--name bert*"):
```
    python3 bert_question_answering_embedding_demo.py
            --vocab=<path_to_model>/vocab.txt
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

## Classifying Documents with Long Texts
Notice that when the original "context" (paragraph text from the url) alone or together with the question do not fit the model input
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
