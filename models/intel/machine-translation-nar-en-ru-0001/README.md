# machine-translation-nar-en-ru-0001

## Use Case and High-Level Description

This is a English-Russian machine translation model based on non-autoregressive Transformer topology.

Tokenization occurs using the SentencePieceBPETokenizer (see the demo code for implementation details) and the enclosed tokenizer_src and tokenizer_tgt folders.

## Specification

| Metric            | Value                 |
|-------------------|-----------------------|
| GOps              | 23.17                 |
| MParams           | 69.29                 |
| Source framework  | PyTorch\*             |

## Accuracy

The quality metrics were calculated on the wmt19-ru-en dataset ("test" split in lower case).

| Metric                    | Value         |
|---------------------------|---------------|
| BLEU                      |        21.6 % |

## Input

name: `tokens`
shape: `1, 192`
description: sequence of tokens (integer values) representing the tokenized sentence.
The sequence structure is as follows (`<s>`, `</s>` and `<pad>` should be replaced by corresponding token IDs as specified by the dictionary):
`<s>` + *tokenized sentence* + `</s>` + (`<pad>` tokens to pad to the maximum sequence length of 192)

## Output

name: `pred`
shape: `1, 192`
description: sequence of tokens (integer values) representing the tokenized translation.
The sequence structure is as follows (`<s>`, `</s>` and `<pad>` should be replaced by corresponding token IDs as specified by the dictionary):
`<s>` + *tokenized sentence* + `</s>` + (`<pad>` tokens to pad to the maximum sequence length of 192)

## Legal Information
[*] Other names and brands may be claimed as the property of others.
