# bert-large-uncased-whole-word-masking-squad-fp32-0001

## Use Case and High-Level Description

This is a BERT-large model pretrained on lower-cased English text using Whole-Word-Masking and fine-tuned on the SQuAD v1.1 training set (93.21% F1, 87.2% EM on the v1.1 dev set). The model performs question answering for English language; the input is a concatenated premise and question for the premise, and the output is the location of the answer to the question inside the premise. For details about the original floating-point model, check out [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805).

Tokenization occurs using the BERT tokenizer (see the demo code for implementation details) and the enclosed `vocab.txt` dictionary file. Input is to be lower-cased before tokenizing.

## Specification

| Metric            | Value                 |
|-------------------|-----------------------|
| GOps              | 246.93                |
| MParams           | 333.96                |
| Source framework  | PyTorch\*             |


## Accuracy

The quality metrics were calculated on the SQuAD v1.1 dataset ("dev" split). Maximum sequence length is 384, maximum query length: 64, doc stride: 128, input is lower-cased.

| Metric                    | Value         |
|---------------------------|---------------|
| F1                        |        93.21% |
| Exact match (EM)          |        87.20% |


## Performance

## Input

Input 0: A `1,384` sequence of tokens (integer values) representing the tokenized premise and question ("input_ids"). The sequence structure is as follows (`[CLS]`, `[SEP]` and `[PAD]` should be replaced by corresponding token IDs as specified by the dictionary):
`[CLS]` + *tokenized question* + `[SEP]` + *tokenized premise of the question* + `[SEP]` + (`[PAD]` tokens to pad to the maximum sequence length of 384)

Input 1: A `1,384` sequence of integer values representing the mask of valid values in the input ("input_mask"). The values of this input are are equal to:
1) `1` at positions corresponding to the `[CLS]` + *tokenized question* + `[SEP]` + *tokenized premise of the question* + `[SEP]` part of the Input 0  (i.e. all positions except those containing the `[PAD]` tokens), and
2) `0` at all other positions

Input 2: A `1,384` sequence of integer values representing the segmentation of the Input 0 into question and premise ("segment_ids"). The values are equal to:
1) `1` at positions corresponding to the *tokenized premise of the question* + `[SEP]` part of the Input 0, and
2) `0` at all other positions

## Output

Output 0: The `1, 384` floating point-valued logit scores, where i-th value corresponds to the log-likelihood of the answer to the question starting at the i-th token position of the input.

Output 1: Same as Output 0, but represents the log-likelihoods of the answer ending at i-th token position.

## Legal Information
[*] Other names and brands may be claimed as the property of others.
