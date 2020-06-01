# bert-small-uncased-whole-word-masking-squad-0001

## Use Case and High-Level Description

This is a small BERT-large like model distilled on SQuAD v1.1 training set from original
bert-large-uncased-whole-word-masking-finetuned-squad provided by the transformes library (https://github.com/huggingface/transformers).
The model performs question answering for English language;
the input is a concatenated premise and question for the premise,
and the output is the location of the answer to the question inside the premise.
For details about the original model, check out
[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805).
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

Input `input_ids`: A `1,384` sequence of tokens (integer values) representing the tokenized premise and question (`input_ids`). The sequence structure is as follows (`[CLS]`, `[SEP]` and `[PAD]` should be replaced by corresponding token IDs as specified by the dictionary):
`[CLS]` + *tokenized question* + `[SEP]` + *tokenized premise of the question* + `[SEP]` + (`[PAD]` tokens to pad to the maximum sequence length of 384)

Input `attention_mask`: A `1,384` sequence of integer values representing the mask of valid values in the input (`input_mask`). The values of this input are are equal to:
1) `1` at positions corresponding to the `[CLS]` + *tokenized question* + `[SEP]` + *tokenized premise of the question* + `[SEP]` part of the `input_ids`  (i.e. all positions except those containing the `[PAD]` tokens), and
2) `0` at all other positions

Input `token_type_ids`: A `1,384` sequence of integer values representing the segmentation of the `input_ids` into question and premise (`segment_ids`). The values are equal to:
1) `1` at positions corresponding to the *tokenized premise of the question* + `[SEP]` part of the `input_ids`, and
2) `0` at all other positions

[CLS] is a special symbol added in front of the question, [SEP] is a special separator token inserted between the question and premise of the question, [PAD] is a special token used to fill the rest of the input to get given input length.

## Output

The outputs of the net are two `1, 384` floating point-valued logit scores vectors that after soft-max operation are probabilities for start and end positions of answer in the premise of the question.

Output `output_s`: The `1, 384` floating point-valued logit scores for start postition.

Output `output_e`: The `1, 384` floating point-valued logit scores for end postition.

## Legal Information
[*] Other names and brands may be claimed as the property of others.
