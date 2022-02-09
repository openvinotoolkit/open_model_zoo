# speech-to-text-en-0001

## Use Case and High-Level Description

Speech To Text (STT) model performs automatic speech recognition. STT's design is based on the QuartzNet architecture,
which is a convolutional model trained with Connectionist Temporal Classification (CTC).
This particular model has 5 QuartzNet blocks each repeated 5 times. The model was trained on Librispeech dataset.

## Specification

| Metric           | Value              |
| ---------------- | ------------------ |
| Type             | Speech recognition |
| GFLOPs           | 0.45               |
| MParams          | 6.9                |
| Source framework | PyTorch\*          |

## Accuracy

| Metric                       | Value |
| ---------------------------- | ----- |
| WER @ Librispeech test-clean | 7.35% |

### Input

Normalized Mel-Spectrogram of 16kHz audio signal, name - `mel`,  shape - `1, 64, 128`, format is `B, N, C`, where:

- `B` - batch size
- `N` - number of mel-spectrogram frequency bins
- `C` - duration

### Output

Per-frame probabilities (after LogSoftmax) for every symbol in the alphabet, name - `preds`,  shape - `1, 128, 256`, output data format is `B, N, C`, where:

- B - batch size
- N - number of audio frames
- C - alphabet size, including the CTC blank symbol

The per-frame probabilities are to be decoded with a CTC decoder.
The alphabet size of 256 tokens in Byte-Pair Encoding format, where.
Begin of sentence: <BOS> - 2
End of sentence: <EOS> - 3
Unknown: <UNK>: 1
Padding symbol: <PAD> - 0
Space symbol: '▁'

## Legal Information
[*] Other names and brands may be claimed as the property of others.
