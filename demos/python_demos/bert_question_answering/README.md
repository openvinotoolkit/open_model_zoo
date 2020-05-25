# Question Answering Python* Demo

This README describes the Question Answering demo application that uses Squad-tuned BERT for inference.

## How It Works

Upon the start-up the demo application reads command line parameters and loads a network to Inference engine and 
an url to the "context" text to search answers for user-provided questions.


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
  -a MAX_ANSWER_TOKEN_NUM, --max_answer_token_num MAX_ANSWER_TOKEN_NUM
                        Optional. Maximum number of tokens in answer
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU
                        is acceptable. Sample will look for a suitable plugin
                        for device specified. Default value is CPU

```

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).


You can use the following command to start inference:
```
    python3 question_answering_demo.py --vocab=<path_to_model>/vocab.txt --model=<path_to_model>/bert_qa.xml --input_names=['input_ids','attention_mask','token_type_ids'] --output_names=['output_s','output_e'] --input=https://en.wikipedia.org/wiki/Speed_of_light
```

## Demo Input

The application reads page by the given url and then iterativly answers questions from console

## Demo Output

The application outputs found answers to the same console.


## See Also
* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)