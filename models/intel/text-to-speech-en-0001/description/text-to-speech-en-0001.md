# text-to-speech-en-0001 (composite)

## Use Case and High-Level Description

This is a speech synthesis composite model that simultaneously reconstruct
mel-spectrogram and wave form from text. The model generate wave form from symbol sequences separated by space.
The model is built on top of the ForwardTacotron and MelGAN frameworks.

## Composite model specification

| Metric                                        | Value     |
|-----------------------------------------------|-----------|
| Word error rate in the part of LJSpeech (based on mozilla-deepspeech-0.6.1) | 33.8% |
| Source framework                              | PyTorch\* |


## Duration prediction model specification

The forward-tacotron-duration-0001 model is a ForwardTacotron-based duration predictor for symbols.

| Metric                                        | Value     |
|-----------------------------------------------|-----------|
| GFlops                                        | 3.54      |
| MParams                                       | 13.81     |

### Inputs

Sequence, name: `input_seq`, shape: [1x128], format: [BxC]
where:
   - B - batch size
   - C - number of symbols in sequence

### Outputs

1. Duration for input symbols, name: `duration`, shape: [1, 128, 1], format [BxCxH]. Contains predicted duration for each of the symbol in sequence.
   - B - batch size
   - C - number of symbols in sequence
   - H - empty dimension
2. Processed embeddings, name: `embeddings`, shape: [1, 128, 512], format [BxCxH]. Contains processed embeddings for each symbol in sequence.
   - B - batch size
   - C - number of symbols in sequence
   - H - height of the intermediate feature map

## Mel-spectrogram regression model specification

The forward-tacotron-regression-0001 model accepts aligned by duration processed embeddings (for example: if duration is [2, 3] and processed embeddings is [[1, 2], [3, 4]], aligned embeddings is [[1, 2], [1, 2], [1,2], [3, 4], [3, 4]]) and produces mel-spectrogram.

| Metric                                        | Value     |
|-----------------------------------------------|-----------|
| GFlops                                        | 3.12      |
| MParams                                       | 3.04      |


### Input

Processed embeddigs aligned by durations, name: `data`, shape: [1x512x512], format: [BxTxC]
where:
   - B - batch size
   - T - time in mel-spectrogram
   - C - processed embedding dimension

### Output

Mel-spectrogram, name: `mel`, shape: [80x512], format: [CxT]
where:
   - T - time in mel-spectrogram
   - C - number of mels in mel-spectrogram


## MelGAN model specification

The melgan-upsample-0001 model is a MelGAN based audio generator.

| Metric                                        | Value |
|-----------------------------------------------|-------|
| GFlops                                        | 80.6  |
| MParams                                       | 12.78 |


### Inputs

Mel-spectrogram, name: `mel`, shape: [1x80x220], format: [BxCxT]
where:
   - B - batch size
   - C - number of symbols in sequence
   - T - time in mel-spectrogram

### Outputs

Audio, name: `audio`, shape: [56320], format: [T]
where:
   - T - time in audio with sampling rate 22050.


## Legal Information
[*] Other names and brands may be claimed as the property of others.
