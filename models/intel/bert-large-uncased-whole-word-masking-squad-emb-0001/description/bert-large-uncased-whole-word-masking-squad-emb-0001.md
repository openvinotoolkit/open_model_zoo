# bert-large-uncased-whole-word-masking-squad-emb-0001

## Use Case and High-Level Description

This is BERT-large model fine-tuned on SQuAD v1.1 training set from original
bert-large-uncased-whole-word-masking provided by the [Transformers](https://github.com/huggingface/transformers) library.
The model performs embeddings for context or question for English language;
the input is a context or question to them,
and the output is the 1024D embedding vectors that allow to find context with answer to the question by simple comparison the context and the question embedding vectors in the 1024D embedding space.
For details about the original model, check out
[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805),
[HuggingFace's Transformers: State-of-the-art Natural Language Processing](https://arxiv.org/abs/1910.03771).

Tokenization occurs using the BERT tokenizer (see the demo code for implementation details) and the enclosed `vocab.txt` dictionary file. Input is to be lower-cased before tokenizing.

## Specification

| Metric            | Value                 |
|-------------------|-----------------------|
| GOps              | 246.93                |
| MParams           | 333.96                |
| Source framework  | PyTorch\*             |

GOps is calculated for for `1, 384` input size that is suitable for long context

## Accuracy

The quality metrics were calculated on the SQuAD v1.1 dataset (`dev` split). Maximum sequence length for context is 384 and for question is 32, input is lower-cased.

| Metric                    | Value         |
|---------------------------|---------------|
| top5                      |        90.5%  |

The top5 is calculated as follow:
1. For each context from  SQuAD v1.1 dev set the context embedding vector is calculated.
2. For each question from  SQuAD v1.1 dev set the question embedding vector is calculated and compared with each previously calculated context embedding vector. If the right context is in top 5 context embedding closest to question embedding then top5_count increased by 1.
3. top5 = top5_count / question_number

## Input

1. Token IDs, name: `input_ids`, shape: `1, 384` for context and `1, 32` for question.
Token IDs is sequence of integer values that is representing the tokenized context or question.
The sequence structure is as follows (`[CLS]`, `[SEP]` and `[PAD]` should be replaced by corresponding token IDs
as specified by the dictionary):
`[CLS]` + *tokenized context or question* + `[SEP]`  + (`[PAD]` tokens to pad to the maximum sequence length of 384 or 32)

2. Input mask, name: `attention_mask`, shape: `1, 384` for context and `1, 32` for question.
Input mask is a sequence of integer values representing the mask of valid values in the input.
The values of this input are equal to:
    * `1` at positions corresponding to the `[CLS]` + *tokenized context or question* + `[SEP]` part of the `input_ids`  (i.e. all positions except those containing the `[PAD]` tokens), and
    * `[PAD]` at all other positions

3. Token types,  name: `token_type_ids`, shape: `1, 384` for context and `1, 32` for question.
Token types is sequence of integer values representing the segmentation of the `input_ids` into question and premise.
The values are equal to:
    * `0` at positions corresponding to the `[CLS]` + *tokenized context or question* + `[SEP]` part of the `input_ids`  (i.e. all positions except those containing the `[PAD]` tokens), and
    * `[PAD]` at all other positions

4. Position indexes,  name: `position_ids`, shape: `1, 384` for context and `1, 32` for question.
Position indexes are sequence of integer values from 0 to 383 (or 31 for question) representing the position index for each input token. The `position_ids` is always the same for any input tokens set

* `[CLS]` is a special symbol added in front of the question.
* `[SEP]` is a special separator token inserted between the question and premise of the question.
* `[PAD]` is a special token used to fill the rest of the input to get given input length (384).

## Output

Embeddings, name: `embedding`, shape `1, 1024`. These vectors can be used to find better context with answer to the question by simple comparing the context embedding vector with question context embedding vector in 1024D embedding space.

## Legal Information
[*] Other names and brands may be claimed as the property of others.

The original `bert-large-uncased-whole-word-masking` model is taken from [Transformers](https://github.com/huggingface/transformers) library, which is distributed under the [Apache License, Version 2.0](https://raw.githubusercontent.com/huggingface/transformers/master/LICENSE).
