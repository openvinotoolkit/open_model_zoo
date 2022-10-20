# machine-translation-nar-en-de-0002

## Use Case and High-Level Description

This is an English-Deutsch machine translation model based on non-autoregressive Transformer topology. The model is [trained](https://github.com/openvinotoolkit/training_extensions/tree/089de2f24667329a58e8560ed4e01ef203e99def/misc/pytorch_toolkit/machine_translation) on internal dataset.

Tokenization occurs using the SentencePieceBPETokenizer (see the demo code for implementation details) and the enclosed tokenizer_src and tokenizer_tgt folders.

## Specification

| Metric            | Value                 |
|-------------------|-----------------------|
| GOps              | 23.19                 |
| MParams           | 77.47                 |
| Source framework  | PyTorch\*             |

## Accuracy

The quality metrics were calculated on the wmt19-en-de dataset ("test" split in lower case).

| Metric                    | Value         |
|---------------------------|---------------|
| BLEU                      |        17.7 % |

Use `accuracy_check [...] --model_attributes <path_to_folder_with_downloaded_model>` to specify the path to additional model attributes. `path_to_folder_with_downloaded_model` is a path to the folder, where the current model is downloaded by [Model Downloader](../../../tools/model_tools/README.md) tool.

## Input

name: `tokens`
shape: `1, 150`
description: sequence of tokens (integer values) representing the tokenized sentence.
The sequence structure is as follows (`<s>`, `</s>` and `<pad>` should be replaced by corresponding token IDs as specified by the dictionary):
`<s>` + *tokenized sentence* + `</s>` + (`<pad>` tokens to pad to the maximum sequence length of 150)

## Output

name: `pred`
shape: `1, 200`
description: sequence of tokens (integer values) representing the tokenized translation.
The sequence structure is as follows (`<s>`, `</s>` and `<pad>` should be replaced by corresponding token IDs as specified by the dictionary):
`<s>` + *tokenized sentence* + `</s>` + (`<pad>` tokens to pad to the maximum sequence length of 150)

## Demo usage

The model can be used in the following demos provided by the Open Model Zoo to show its capabilities:

* [Machine Translation Python\* Demo](../../../demos/machine_translation_demo/python/README.md)

## Legal Information
[*] Other names and brands may be claimed as the property of others.
