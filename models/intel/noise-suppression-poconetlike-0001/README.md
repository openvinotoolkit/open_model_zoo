# noise-suppression-poconetlike-0001

## Use Case and High-Level Description

This is a PoCoNet like model for noise suppression to make speech cleaner.
The model is based on [PoCoNet](https://arxiv.org/abs/2008.04470) architecure and trained on [DNS-Challenge dataset](https://github.com/microsoft/DNS-Challenge/blob/master/README.md) [paper](https://arxiv.org/abs/2101.01902).
The network works with mono audio sampled on 16kHz.
The audio processed iterative by patches with 2048 size.
On each iteration it takes 2048 (128ms) samples as input and returns 2048 (128ms) samples as output with 640 (40ms) samples delay.
In addition the network required 50 state tensors to make processing seamless.
On the first iteration these state tensors have to be filled with 0.
On the consequences iterations theses tensors have to be taken from corresponding outputs of previous iteration.
You can try [Noise Suppression Python\* Demo](../../../demos/noise_suppression_demo/python/README.md) to see how it works.

## Specification

to process 2048 samples that is 128ms for 16kHz

| Metric            | Value                 |
|-------------------|-----------------------|
| GOps              | 1.2                   |
| MParams           | 7.22                  |
| Source framework  | PyTorch\*             |
## Accuracy

The [SISDR](https://arxiv.org/abs/1811.02508) quality metric was calculated on the 100 [dev test synthetic speech clips from DNS-Challenge 2021 dataset](https://github.com/microsoft/DNS-Challenge/blob/master/README-DNS3.md).


| Metric                          | Value         |
|---------------------------------|---------------|
| SISDR for input noisy signal    |    11.73   dB |
| SISDR for output cleaned signal |    20.54   dB |
| SISDR increase                  |    +8.81   dB |


## Input

Sequence patch, name: `input`, shape: `1, 2048`, format: `B, T`, where:

 - `B` - batch size
 - `T` - number of samples in patch

input states, names: `inp_state_*`, should be filled by corresponding `out_state_*` from previous step

## Output

Sequence patch, name: `output`, shape: `1, 2048`, format: `B, T`, where:

 - `B` - batch size
 - `T` - number of samples in patch
Note: The output patch is "shifted" by 640 (40ms) samples in time. So output[0,i] sample is synced with input[0,i-640] sample

output states, names: `out_state_*`, should be used to fill corresponding `inp_state_*` on next step

## Demo usage

The model can be used in the following demos provided by the Open Model Zoo to show its capabilities:

* [Noise Suppression C++\* Demo](../../../demos/noise_suppression_demo/cpp/README.md)
* [Noise Suppression Python\* Demo](../../../demos/noise_suppression_demo/python/README.md)

## Legal Information
[*] Other names and brands may be claimed as the property of others.
