# bert-small-uncased-whole-word-masking-squad-0001

## Use Case and High-Level Description

This is a small BERT-large like model distilled on SQuAD v1.1 training set from original
bert-large-uncased-whole-word-masking-finetuned-squad provided by the [Transformers](https://github.com/huggingface/transformers) library.
The model performs question answering for English language;
the input is a concatenated premise and question for the premise,
and the output is the location of the answer to the question inside the premise.
For details about the original model, check out
[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805),
[HuggingFace's Transformers: State-of-the-art Natural Language Processing](https://arxiv.org/abs/1910.03771).

Tokenization occurs using the BERT tokenizer (see the demo code for implementation details) and the enclosed `vocab.txt` dictionary file. Input is to be lower-cased before tokenizing.

## Specification

| Metric            | Value                 |
|-------------------|-----------------------|
| GOps              | 23.9                  |
| MParams           | 57.94                 |
| Source framework  | PyTorch\*             |

## Accuracy

The quality metrics were calculated on the SQuAD v1.1 dataset ("dev" split). Maximum sequence length is 384, input is lower-cased.

| Metric                    | Value         |
|---------------------------|---------------|
| F1                        |        91.57% |
| Exact match (EM)          |        85.04% |

## Performance

## Input

1. Token IDs, name: `input_ids`, shape: [1x384].
Token IDs is sequence of integer values that is representing the tokenized premise and question.
The sequence structure is as follows (`[CLS]`, `[SEP]` and `[PAD]` should be replaced by corresponding token IDs
as specified by the dictionary):
`[CLS]` + *tokenized question* + `[SEP]` + *tokenized premise of the question* + `[SEP]` + (`[PAD]` tokens to pad to the maximum sequence length of 384)

2. Input mask, name: `attention_mask`, shape: [1x384].
Input mask is a sequence of integer values representing the mask of valid values in the input.
The values of this input are equal to:
    * `1` at positions corresponding to the `[CLS]` + *tokenized question* + `[SEP]` + *tokenized premise of the question* + `[SEP]` part of the `input_ids`  (i.e. all positions except those containing the `[PAD]` tokens), and
    * `0` at all other positions

3. Token types,  name: `token_type_ids`, shape: [1x384].
Token types is sequence of integer values representing the segmentation of the `input_ids` into question and premise.
The values are equal to:
    * `1` at positions corresponding to the *tokenized premise of the question* + `[SEP]` part of the `input_ids`, and
    * `0` at all other positions

* `[CLS]` is a special symbol added in front of the question.
* `[SEP]` is a special separator token inserted between the question and premise of the question.
* `[PAD]` is a special token used to fill the rest of the input to get given input length (384).

## Output

The outputs of the net are two `1, 384` floating point-valued logit scores vectors that after soft-max operation are probabilities for start and end positions of answer in the premise of the question.

1. Start position: name: `output_s`, shape: [1x384].
Start position is floating point-valued logit scores for start postition.

2. End position: name: `output_e`, shape: [1x384].
End position is floating point-valued logit scores for end postition.

## Legal Information
[*] Other names and brands may be claimed as the property of others.

The original `bert-large-uncased-whole-word-masking-finetuned-squad` model is taken from [Transformers](https://github.com/huggingface/transformers) library, which is distributed under the [Apache License, Version 2.0](https://raw.githubusercontent.com/huggingface/transformers/master/LICENSE). 
