# bert-base-ner

## Use Case and High-Level Description

`bert-base-ner` is a fine-tuned BERT model that is ready to use for Named Entity Recognition and achieves state-of-the-art performance for the NER task. It has been trained to recognize four types of entities: location (LOC), organizations (ORG), person (PER) and Miscellaneous (MISC).

Specifically, this model is a bert-base-cased model that was fine-tuned on the English version of the standard [CoNLL-2003 Named Entity Recognition](https://www.aclweb.org/anthology/W03-0419) dataset.
For details about the original model, check out
[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805),
[HuggingFace's Transformers: State-of-the-art Natural Language Processing](https://arxiv.org/abs/1910.03771) papers and [repository](https://github.com/huggingface/transformers).

Tokenization occurs using the BERT tokenizer (see the demo code for implementation details) and the enclosed `vocab.txt` dictionary file.

## Specification

| Metric            | Value                 |
|-------------------|-----------------------|
| GOps              | 22.3874               |
| MParams           | 107.4319              |
| Source framework  | PyTorch\*             |


## Accuracy

The quality metric was calculated on CONLL-2003 Named Entity Recognition dataset (dev set). Input sequences were padded to 128 symbols.

| Metric                    | Value         |
|---------------------------|---------------|
| F1                        |        94.45% |

## Input

### Original model

1. Token IDs, name: `input_ids`, shape: `1, 128`.
Token IDs is sequence of integer values that is representing the tokenized input sentence.
The sequence structure is as follows (`[CLS]` and `[SEP]`should be replaced by corresponding token IDs
as specified by the dictionary):
`[CLS]` + *tokenized text* + `[SEP]` + `0` (for padding to sequence length 128]

2. Input mask, name: `attention_mask`, shape: `1, 128`.
Input mask is a sequence of integer values representing the mask of valid values in the input.
The values of this input are equal to:
    * `1` at positions corresponding to the `[CLS]` + *tokenized text* + `[SEP]` part of the `input_ids`  (i.e. all positions except those containing the padding), and
    * `0` at all other positions

3. Token types,  name: `token_type_ids`, shape: `1, 128`.
Token types is sequence of integer values representing the segmentation of the `input_ids`.
The values are equal to `0` at all other positions (all text belongs to single segment)

* `[CLS]` is a special symbol added in front of the text.
* `[SEP]` is a special separator added at the end of the text.

### Converted model

 Converted model has the same inputs like in original.

## Output

### Original model

Token classifier, name: `output`, shape: `1, 128, 9`
floating point-valued logit scores vectors that represents probability of belonging each token to 9 classes:

| Abbreviation| Description                                                                  |
| ----------- | ---------------------------------------------------------------------------- |
| O           | Outside of a named entity                                                    |
| B-MIS       | Beginning of a miscellaneous entity right after another miscellaneous entity |
| I-MIS       | Miscellaneous entity                                                         |
| B-PER       | Beginning of a person’s name right after another person’s name               |
| I-PER       | Person’s name                                                                |
| B-ORG       | Beginning of an organisation right after another organisation                |
| I-ORG       | Organisation                                                                 |
| B-LOC       | Beginning of a location right after another location                         |
| I-LOC       | Location                                                                     |

### Converted model

Converted model has the same output like original.

## Download a Model and Convert it into Inference Engine Format

You can download models and if necessary convert them into Inference Engine format using the [Model Downloader and other automation tools](../../../tools/downloader/README.md) as shown in the examples below.

An example of using the Model Downloader:
```
python3 <omz_dir>/tools/downloader/downloader.py --name <model_name>
```

An example of using the Model Converter:
```
python3 <omz_dir>/tools/downloader/converter.py --name <model_name>
```

## Legal Information

The original model is distributed under [Apache License, Version 2.0](https://raw.githubusercontent.com/huggingface/transformers/master/LICENSE).
