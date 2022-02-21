# text-to-speech-en-0002 (composite)

## Use Case and High-Level Description

This is a mel-spectrogram synthesis composite model. The model generates mel-spectrogram from symbol sequences separated by space.

## Composite model specification

| Metric                                        | Value     |
|-----------------------------------------------|-----------|
| Source framework                              | PyTorch\* |

## Encoder specification

The text-to-speech-en-0002-encoder model is a encoder part of GAN based network. This part creates intermediate representation for text and predicts duaration for every input symbol.

| Metric                                        | Value     |
|-----------------------------------------------|-----------|
| GFlops                                        | 2.44      |
| MParams                                       | 9.59      |

### Inputs

1. Sequence, name: `seq`, shape: `1, 128`, format: `B,C`, where:

    - `B` - batch size
    - `C` - number of symbols in sequence

2. Lengths for generating mask for input sequence, name: `seq_len`, shape: `1`, format: `B`, where:

    - `B` - batch size

### Outputs

1. Feature map for building attention map , name: `x_m`, shape: `1, 80, 128`, format `B, C, H`. Contains feature map for computing alignment between text and mel-spectrogram during training procedure.

    - `B` - batch size
    - `C` - number of channels in feature map
    - `H` - number of symbols in sequence

2. Feature map for decoder, name: `x_res`, shape: `1, 128, 128`, format `B, C, H`.

    - `B` - batch size
    - `C` - number of channels in feature map
    - `H` - number of symbols in sequence

3. Logarithm of duration for input symbols, name: `logw`, shape: `1, 1, 128`, format `B, C, H`. Contains logarithm of predicted duration for each of the symbol in sequence.

    - `B` - batch size
    - `C` - number of channels in feature map
    - `H` - number of symbols in sequence

4. Mask for computation of attention map, name: `x_mask`, shape: `1, 1, 128`, format `B, C, H`.

    - `B` - batch size
    - `C` - number of channels in feature map
    - `H` - number of symbols in sequence


## Decoder specification

The text-to-speech-en-0002-decoder model accepts aligned by duration processed embeddings (for example: if duration is [2, 3] and processed embeddings is [[1, 2], [3, 4]], aligned embeddings is [[1, 2], [1, 2], [1,2], [3, 4], [3, 4]]) and produces mel-spectrogram.

| Metric                                        | Value     |
|-----------------------------------------------|-----------|
| GFlops                                        | 2.32      |
| MParams                                       | 4.44      |

### Inputs

1. Processed embeddigs aligned by durations, name: `z`, shape: `1, 128, 256`, format: `B, C, T`, where:

    - `B` - batch size
    - `T` - time in mel-spectrogram
    - `C` - processed embedding dimension

2. Mask for `z` by time dimension, name: `z_mask`, shape: `1, 1, 256`, format: `B, D, T`, where:

    - `B` - batch size
    - `D` - extra dimension for multiplication
    - `T` - time in mel-spectrogram

### Output

Mel-spectrogram, name: `mel`, shape: `1, 80, 256`, format: `B, C, T`, where:

    - `B` - batch size
    - `T` - time in mel-spectrogram
    - `C` - number of rows in mel-spectrogram

## Legal Information
[*] Other names and brands may be claimed as the property of others.
