# noise-suppression-denseunet-ll-0001

## Use Case and High-Level Description

This is a model for noise suppression to make speech cleaner.
The model architecture is similar to [PoCoNet](https://arxiv.org/abs/2008.04470), [Channel-Attention Dense U-Net](https://arxiv.org/abs/2001.11542) but without multi head attensions (MHA) to decrease model complexity and increse processing speed.
Also to reduce processed patch size and achive small delay the pyrmid structure along time axis is reduced and convolutions with dilation along time axis are used. 
The model was trained on [DNS-Challenge dataset](https://github.com/microsoft/DNS-Challenge/blob/master/README.md) [paper](https://arxiv.org/abs/2101.01902), .
The network works with mono audio sampled on 16kHz.
The audio is processed iteratively by patches with 128 size.
On each iteration it takes 128 (8ms) samples as input and returns 128 (8ms) samples as output with 384 (24ms) samples delay.
In addition the network required 39 state tensors to make processing seamless.
On the first iteration these state tensors have to be filled with 0.
On the consequences iterations these tensors have to be taken from corresponding outputs of previous iteration.
You can try [Noise Suppression CPP\* Demo](../../../demos/noise_suppression_demo/cpp/README.md) to see how it works.

## Specification

to process 128 samples that is 8ms for 16kHz

| Metric            | Value                 |
|-------------------|-----------------------|
| GOps              | 0.2                   |
| MParams           | 4.2                   |
| Source framework  | PyTorch\*             |
## Accuracy

The [SISDR](https://arxiv.org/abs/1811.02508) quality metric was calculated on the 100 [dev test synthetic speech clips from DNS-Challenge dataset](https://github.com/microsoft/DNS-Challenge/tree/icassp2021-final/datasets/ICASSP_dev_test_set/track_1/synthetic).


| Metric                          | Value         |
|---------------------------------|---------------|
| SISDR for input noisy signal    |    11.7    dB |
| SISDR for output cleaned signal |    20.0    dB |
| SISDR increase                  |    +8.3    dB |


## Input

Sequence patch, name: `input`, shape: `1, 128`, format: `B, T`, where:

 - `B` - batch size
 - `T` - number of samples in patch

input states, names: `inp_state_*`, should be filled by corresponding `out_state_*` from previous step

## Output

Sequence patch, name: `output`, shape: `1, 128`, format: `B, T`, where:

 - `B` - batch size
 - `T` - number of samples in patch
Note: The output patch is "shifted" by 384 (24ms) samples in time. So output[0,i] sample is synced with input[0,i-384] sample

output states, names: `out_state_*`, should be used to fill corresponding `inp_state_*` on next step

## Legal Information
[*] Other names and brands may be claimed as the property of others.
