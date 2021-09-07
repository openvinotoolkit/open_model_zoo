# quartznet-15x5-en

## Use Case and High-Level Description

QuartzNet model performs automatic speech recognition. QuartzNet's design is based on the Jasper architecture,
which is a convolutional model trained with Connectionist Temporal Classification (CTC) loss.
This particular model has 15 Jasper blocks each repeated 5 times. The model was trained in NeMo on multiple datasets:
LibriSpeech, Mozilla Common Voice, WSJ, Fisher, Switchboard, and NSC Singapore English.
For details see [repository](https://github.com/NVIDIA/NeMo), [paper](https://arxiv.org/pdf/1910.10261.pdf).

## Specification

| Metric           | Value              |
| ---------------- | ------------------ |
| Type             | Speech recognition |
| GFLOPs           | 2.4195             |
| MParams          | 18.8857            |
| Source framework | PyTorch\*          |

## Accuracy

| Metric                       | Value |
| ---------------------------- | ----- |
| WER @ Librispeech test-clean | 3.86% |

### Input

#### Original model

Normalized Mel-Spectrogram of 16kHz audio signal, name - `audio_signal`,  shape - `1, 64, 128`, format is `B, N, C`, where:

- `B` - batch size
- `N` - number of mel-spectrogram frequency bins
- `C` - duration

#### Converted model

The converted model has the same parameters as the original model.

### Output

#### Original model

Per-frame probabilities (after LogSoftmax) for every symbol in the alphabet, name - `output`,  shape - `1, 64, 29`, output data format is `B, N, C`, where:

- B - batch size
- N - number of audio frames
- C - alphabet size, including the CTC blank symbol

The per-frame probabilities are to be decoded with a CTC decoder.
The alphabet is: 0 = space, 1...26 = "a" to "z", 27 = apostrophe, 28 = CTC blank symbol. Example is provided [here](../../../demos/speech_recognition_deepspeech_demo/python/default_alphabet_example.conf).

#### Converted model

The converted model has the same parameters as the original model.

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

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/NVIDIA/NeMo/main/LICENSE).
A copy of the license is provided in [APACHE-2.0.txt](../licenses/APACHE-2.0.txt).
