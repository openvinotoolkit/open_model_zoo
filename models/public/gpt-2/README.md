# gpt-2

## Use Case and High-Level Description

The `gpt-2` model is a one of Generative Pre-trained Transformer (GPT) model family, pre-trained on a very large corpus of English data in a self-supervised fashion. The GPT architecture implements a deep neural network, specifically a transformer model, which uses attention in place of previous recurrence- and convolution-based architectures. Attention mechanisms allow the model to selectively focus on segments of input text it predicts to be the most relevant. GPT-2 is trained with a simple objective: predict the next word, given all of the previous words within some text.

More details provided in the [paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), [repository](https://github.com/huggingface/transformers) and [model card](https://huggingface.co/gpt2).

## Specification

| Metric            | Value            |
|-------------------|------------------|
| Type              | Text Prediction  |
| GFlops            | 293.0489         |
| MParams           | 175.6203         |
| Source framework  | PyTorch\*        |

GFlops calculated for `1, 1024` input shape, that is suitable for long context

## Accuracy

[Perplexity](https://en.wikipedia.org/wiki/Perplexity) obtained on [WikiText-2 raw character level data](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) dataset for converted model.

| Metric     | Value  |
| ---------- | ------ |
| Perplexity | 29.00% |

## Input

### Original model

Token ids, name: `input`, dynamic shape in the format `B, L`, where:

- `B` - batch size
- `L` - sequence length

### Converted model

Token ids, name: `input`, dynamic shape in the format `B, L`, where:

- `B` - batch size
- `L` - sequence length

## Output

### Original model

Prediction scores of language modeling head, name: `output`, dynamic shape `B, L, 50257` in the format `B, L, S`, where:

- `B` - batch size
- `L` - sequence length
- `S` - vocab size

### Converted model

Prediction scores of language modeling head, name: `output`, dynamic shape `B, L, 50257` in the format `B, L, S`, where:

- `B` - batch size
- `L` - sequence length
- `S` - vocab size

## Download a Model and Convert it into OpenVINO™ IR Format

You can download models and if necessary convert them into OpenVINO™ IR format using the [Model Downloader and other automation tools](../../../tools/model_tools/README.md) as shown in the examples below.

An example of using the Model Downloader:
```
omz_downloader --name <model_name>
```

An example of using the Model Converter:
```
omz_converter --name <model_name>
```

## Demo usage

The model can be used in the following demos provided by the Open Model Zoo to show its capabilities:

* [GPT-2 Text Prediction Python\* Demo](../../../demos/gpt2_text_prediction_demo/python/README.md)

## Legal Information

The original model is distributed under the
[mit License](https://huggingface.co/gpt2).
