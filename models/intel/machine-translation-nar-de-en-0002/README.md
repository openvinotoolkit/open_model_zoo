# machine-translation-nar-de-en-0002

## Use Case and High-Level Description

This is a Deutsch-English machine translation model based on non-autoregressive Transformer topology.

Tokenization occurs using the SentencePieceBPETokenizer (see the demo code for implementation details) and the enclosed tokenizer_src and tokenizer_tgt folders.

## Specification

| Metric            | Value                 |
|-------------------|-----------------------|
| GOps              | 23.19                 |
| MParams           | 77.47                 |
| Source framework  | PyTorch\*             |

## Accuracy

The quality metrics were calculated on the wmt19-en-de dataset (`test` split in lower case).

| Metric                    | Value         |
|---------------------------|---------------|
| BLEU                      |        21.4 % |

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
