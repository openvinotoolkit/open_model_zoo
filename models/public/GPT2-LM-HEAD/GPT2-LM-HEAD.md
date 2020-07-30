# GPT2-LM-HEAD

## Use Case and High-Level Description

GPT2-LM-HEAD: Transformer-based language model for text generation. For details see the [paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), [original repository](https://github.com/huggingface/transformers) and repository with [converted model](https://github.com/onnx/models/tree/master/text/machine_comprehension/gpt-2).

## Specification

| Metric            | Value   |
|-------------------|---------|
| Accuracy          |         |
| GFlops            |         |
| MParams           |         |
| Source framework  | ONNX\*  |

## Performance

## Input

### Original model

Image, name - `input1` , dynamic shape, format [Bx1xL],
   where:

- B - batch size
- L - sequence length

### Converted model

Image, name - `input1` , shape - [1x1x1024], format [Bx1xC],
   where:

- B - batch size
- L - sequence length

## Output

### Original model

1. Prediction scores of language modeling head , name: `MatMul_2533`, shape - [1x1x8x50257]. Presented in format [Bx1xLxS],
    where:

- B - batch size
- L - sequence length
- S - vocab size

2. Pre-computed hidden states, names: `Concat_2398,Concat_2191,Concat_1984,Concat_1777,Concat_1570,Concat_1363,Concat_1156,Concat_949,Concat_742,Concat_535,Concat_328,Concat_121`, shape - [2x1x12x8x64] in format [Bx1xNxLx?],
    where:

- B - batch size
- N - num heads
- L - sequence length

### Converted model

Prediction scores of language modeling head , name: `MatMul_2533`, shape - [1x1x1024x50257]. Presented in format [Bx1xLxS],
    where:

    - B - batch size
    - L - sequence length
    - S - vocab size

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/huggingface/transformers/master/LICENSE).
A copy of the license is provided in [APACHE-2.0.txt](../licenses/APACHE-2.0.txt).

[*] Other names and brands may be claimed as the property of others.
