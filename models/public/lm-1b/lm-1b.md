# lm-1b

## Use Case and High-Level Description

The `lm-1b` model is used to language prediction. It's a model hybrid between character CNN, a large and deep LSTM, and a specific Softmax architecture. The `lm-1b` model is pretrained on the One Billion Word Benchmark dataset, and get the best perplexity thus far. For details about this model, check out the paper (http://arxiv.org/abs/1602.02410).

The model inputs include embeded characters whose shape is [50] and placeholders whose shape is [1,9216].

The model output for `lm-1b` is the language prediction and its perplexity.

For more details see repository <https://github.com/tensorflow/models/tree/master/research/lm_1b>

## Example

## Specification

| Metric            | Value              |
|-------------------|--------------------|
| Type              | Language Prediction|
| Source framework  | TensorFlow\*       |

## Accuracy

Test on the test dataset `lm-1b-dataset`, At convergence, the perplexity should be around 30.
See [https://github.com/tensorflow/models/tree/master/research/lm_1b]

## Performance

## Input

### Original model

  --ckpt: Checkpoint directory used to fill model values.
  --input_data: Input data files for eval model.
  --max_eval_steps: Maximum mumber of steps to run "eval" mode.
  --max_sample_words: Sampling stops either when </S> is met or this number of steps has passed.
  --mode: One of [sample, eval, dump_emb, dump_lstm_emb]. "sample" mode samples future word predictions, using FLAGS.prefix as prefix (prefix could be left empty). "eval" mode calculates perplexity of the FLAGS.input_data. "dump_emb" mode dumps word and softmax embeddings to FLAGS.save_dir. embeddings are dumped in the same order as words in vocabulary. All words in vocabulary are dumped.dump_lstm_emb dumps lstm embeddings of FLAGS.sentence to FLAGS.save_dir.
  --num_samples: Number of samples to generate for the prefix.
  --pbtxt: GraphDef proto text file used to construct model structure.
  --prefix: Used for "sample" mode to predict next words.
  --save_dir: Used for "dump_emb" mode to save word embeddings.
  --sentence: Used as input for "dump_lstm_emb" mode.
  --vocab_file: Vocabulary file.

### Converted model

- `i` - The prefix used for word predictions
- `v` - The vocab file

## Output

### Original model

In the sample mode, the model output a prediction word.
In the eval mode, the model output the perplexity of each sentence.

### Converted model

The output of converted model is similar to the one of original model.

## Legal Information

The original model is distributed under the following
[license](https://github.com/tensorflow/models/blob/master/research/lm_1b/README.md)

```
```
