# text-to-speech-en-multi-0001 (composite)

## Use Case and High-Level Description

This is a speech synthesis composite model that simultaneously reconstructs
mel-spectrogram and wave form from text. The model generates wave form from symbol sequences separated by space for
forty speakers. The speaker voice characteristics are represented by vector of two numbers.
The model is built on top of the modified ForwardTacotron and modified MelGAN frameworks.

## Composite model specification

| Metric                                        | Value     |
|-----------------------------------------------|-----------|
| Source framework                              | PyTorch\* |

## Duration prediction model specification

The text-to-speech-en-multi-0001-duration-prediction model is a ForwardTacotron-based duration predictor for symbols.

| Metric                                        | Value     |
|-----------------------------------------------|-----------|
| GFlops                                        | 28.75     |
| MParams                                       | 26.18     |

### Inputs

1. Sequence, name: `input_seq`, shape: `1, 512`, format: `B,C`, where:

    - `B` - batch size
    - `C` - number of symbols in sequence

2. Mask for input sequence, name: `input_mask`, shape: `1, 1, 512`, format: `B, D, C`, where:

    - `B` - batch size
    - `D` - extra dimension for multiplication
    - `C` - number of symbols in sequence

3. Mask for relative position representation in attention, name: `pos_mask`, shape: `1, 1, 512, 512`, format: `B, D, C, C`, where:

    - `B` - batch size
    - `D` - extra dimension for multiplication
    - `C` - number of symbols in sequence

4. Vector for representing the speaker voice embedding, name: `speaker_embedding`, shape: `1, 2`, format: `B, D`, where:

    - `B` - batch size
    - `D` - size of the embedding vector

### Outputs

1. Duration for input symbols, name: `duration`, shape: `1, 512, 1`, format `B, C, H`. Contains predicted duration for each of the symbol in sequence.

    - `B` - batch size
    - `C` - number of symbols in sequence
    - `H` - empty dimension

2. Processed embeddings, name: `embeddings`, shape: `1, 512, 256`, format `B, C, H`. Contains processed embeddings for each symbol in sequence.

    - `B` - batch size
    - `C` - number of symbols in sequence
    - `H` - height of the intermediate feature map

## Mel-spectrogram regression model specification

The text-to-speech-en-multi-0001-regression model accepts aligned by duration processed embeddings (for example: if duration is [2, 3] and processed embeddings is [[1, 2], [3, 4]], aligned embeddings is [[1, 2], [1, 2], [1,2], [3, 4], [3, 4]]) and produces mel-spectrogram.

| Metric                                        | Value     |
|-----------------------------------------------|-----------|
| GFlops                                        | 7.81      |
| MParams                                       | 5.12      |

### Inputs

1. Processed embeddigs aligned by durations, name: `data`, shape: `1, 512, 256`, format: `B, T, C`, where:

    - `B` - batch size
    - `T` - time in mel-spectrogram
    - `C` - processed embedding dimension

2. Mask for `data` by time dimension, name: `data_mask`, shape: `1, 1, 512`, format: `B, D, T`, where:

    - `B` - batch size
    - `D` - extra dimension for multiplication
    - `T` - time in mel-spectrogram

3. Mask for relative position representation in attention, name: `pos_mask`, shape: `1, 1, 512, 512`, format: `B, D, C, C`, where:

    - `B` - batch size
    - `D` - extra dimension for multiplication
    - `C` - number of symbols in sequence

4. Vector for representing the speaker voice embedding, name: `speaker_embedding`, shape: `1, 2`, format: `B, D`, where:

    - `B` - batch size
    - `D` - size of the embedding vector

### Output

Mel-spectrogram, name: `mel`, shape: `80, 512`, format: `C, T`, where:

- `T` - time in mel-spectrogram
- `C` - number of rows in mel-spectrogram

## Audio generation model specification

The text-to-speech-en-multi-0001-generation model is a MelGAN based audio generator.

| Metric                                        | Value |
|-----------------------------------------------|-------|
| GFlops                                        | 48.38 |
| MParams                                       | 12.77 |

### Inputs

Mel-spectrogram, name: `mel`, shape: `1, 80, 128`, format: `B, C, T`, where:

- `B` - batch size
- `C` - number of rows in mel-spectrogram
- `T` - time in mel-spectrogram

### Outputs

Audio, name: `audio`, shape: `32768`, format: `T`, where:

- `T` - time in audio with sampling rate 22050 (~1.5 sec).

## Legal Information
[*] Other names and brands may be claimed as the property of others.
