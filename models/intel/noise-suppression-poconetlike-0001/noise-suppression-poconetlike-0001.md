# noise-suppression-poconetlike-0001

## Use Case and High-Level Description

This is a PoCoNet like model for noise suppression.
The model is based on [PoCoNet](https://arxiv.org/abs/2008.04470) architecure and trained on clean and noise samples from DNS-chalange dataset [github](https://github.com/microsoft/DNS-Challenge/blob/master/README.md) [paper](https://arxiv.org/abs/2101.01902).
The network works with mono audio sampled on 16kHz.
The audio processed iterative by patches with 2048 size.
On each iteration it takes 2048 (128ms) samples as input and returns 2048 (128ms) samples as output with 640 (40ms) samples delay.
In addition the network required 52 state tensors to make processing seamless.
On the first iteration these state tensors have to be filled by 0.
On the consequences iterations theses tensors have to be taken from corresponding outputs of previous iteration.
You can try [demo](../../../demos/noise_suppresion/python) to see how it works.

## Specification

to process 2048 samples that is 128ms for 16kHz

| Metric            | Value                 |
|-------------------|-----------------------|
| GOps              | 1.2                   |
| MParams           | 7.22                  |
| Source framework  | PyTorch\*             |



## Accuracy

The [SISDR](https://arxiv.org/abs/1811.02508) quality metric was calculated on the 100 dev test synthetic speech clips from DNS-chalange dataset [github](https://github.com/microsoft/DNS-Challenge/tree/icassp2021-final/datasets/ICASSP_dev_test_set/track_1/synthetic).


| Metric                    | Value         |
|---------------------------|---------------|
| SISDR OUT                 |    20.23   dB |
| SISDR DIFF                |     8.49   dB |


## Input

Sequence patch, name: `input`, shape: `1, 2048`, format: `B, T`
where:
   - B - batch size
   - T - number of samples in patch

input state, name: `inp_state_000`, shape: `1, 128`

input state, name: `inp_state_001`, shape: `1, 2, 129, 2`

input state, name: `inp_state_002`, shape: `1, 18, 129, 2`

input state, name: `inp_state_003`, shape: `1, 34, 129, 2`

input state, name: `inp_state_004`, shape: `129, 32, 32`

input state, name: `inp_state_005`, shape: `129, 32, 32`

input state, name: `inp_state_006`, shape: `1, 32, 64, 2`

input state, name: `inp_state_007`, shape: `1, 48, 64, 2`

input state, name: `inp_state_008`, shape: `1, 64, 64, 2`

input state, name: `inp_state_009`, shape: `64, 64, 22`

input state, name: `inp_state_010`, shape: `64, 64, 22`

input state, name: `inp_state_011`, shape: `1, 64, 32, 2`

input state, name: `inp_state_012`, shape: `1, 96, 32, 2`

input state, name: `inp_state_013`, shape: `1, 128, 32, 2`

input state, name: `inp_state_014`, shape: `32, 128, 16`

input state, name: `inp_state_015`, shape: `32, 128, 16`

input state, name: `inp_state_016`, shape: `1, 128, 16, 2`

input state, name: `inp_state_017`, shape: `1, 192, 16, 2`

input state, name: `inp_state_018`, shape: `1, 256, 16, 2`

input state, name: `inp_state_019`, shape: `16, 256, 11`

input state, name: `inp_state_020`, shape: `16, 256, 11`

input state, name: `inp_state_021`, shape: `1, 256, 8, 2`

input state, name: `inp_state_022`, shape: `1, 384, 8, 2`

input state, name: `inp_state_023`, shape: `1, 512, 8, 2`

input state, name: `inp_state_024`, shape: `8, 256, 8`

input state, name: `inp_state_025`, shape: `8, 256, 8`

input state, name: `inp_state_026`, shape: `1, 512, 16, 2`

input state, name: `inp_state_027`, shape: `1, 256, 16, 2`

input state, name: `inp_state_028`, shape: `1, 384, 16, 2`

input state, name: `inp_state_029`, shape: `1, 512, 16, 2`

input state, name: `inp_state_030`, shape: `16, 128, 11`

input state, name: `inp_state_031`, shape: `16, 128, 11`

input state, name: `inp_state_032`, shape: `1, 256, 32, 2`

input state, name: `inp_state_033`, shape: `1, 128, 32, 2`

input state, name: `inp_state_034`, shape: `1, 192, 32, 2`

input state, name: `inp_state_035`, shape: `1, 256, 32, 2`

input state, name: `inp_state_036`, shape: `32, 64, 16`

input state, name: `inp_state_037`, shape: `32, 64, 16`

input state, name: `inp_state_038`, shape: `1, 128, 64, 2`

input state, name: `inp_state_039`, shape: `1, 64, 64, 2`

input state, name: `inp_state_040`, shape: `1, 96, 64, 2`

input state, name: `inp_state_041`, shape: `1, 128, 64, 2`

input state, name: `inp_state_042`, shape: `64, 32, 22`

input state, name: `inp_state_043`, shape: `64, 32, 22`

input state, name: `inp_state_044`, shape: `1, 64, 129, 2`

input state, name: `inp_state_045`, shape: `1, 32, 129, 2`

input state, name: `inp_state_046`, shape: `1, 48, 129, 2`

input state, name: `inp_state_047`, shape: `1, 64, 129, 2`

input state, name: `inp_state_048`, shape: `1, 129, 4`

input state, name: `inp_state_049`, shape: `1, 129, 4`

input state, name: `inp_state_050`, shape: `1, 129, 1`

input state, name: `inp_state_051`, shape: `1, 129, 1`

## Output

Sequence patch, name: `output`, shape: `1, 2048`, format: `B, T`
where:
   - B - batch size
   - T - number of samples in patch
Note: The ouput patch is "shifted" by 640 (40ms) samples in time. So output[0,i] sample is synced with input[0,i-640] sample

output state, name: `out_state_000`, shape: `1, 128`

output state, name: `out_state_001`, shape: `1, 2, 129, 2`

output state, name: `out_state_002`, shape: `1, 18, 129, 2`

output state, name: `out_state_003`, shape: `1, 34, 129, 2`

output state, name: `out_state_004`, shape: `129, 32, 32`

output state, name: `out_state_005`, shape: `129, 32, 32`

output state, name: `out_state_006`, shape: `1, 32, 64, 2`

output state, name: `out_state_007`, shape: `1, 48, 64, 2`

output state, name: `out_state_008`, shape: `1, 64, 64, 2`

output state, name: `out_state_009`, shape: `64, 64, 22`

output state, name: `out_state_010`, shape: `64, 64, 22`

output state, name: `out_state_011`, shape: `1, 64, 32, 2`

output state, name: `out_state_012`, shape: `1, 96, 32, 2`

output state, name: `out_state_013`, shape: `1, 128, 32, 2`

output state, name: `out_state_014`, shape: `32, 128, 16`

output state, name: `out_state_015`, shape: `32, 128, 16`

output state, name: `out_state_016`, shape: `1, 128, 16, 2`

output state, name: `out_state_017`, shape: `1, 192, 16, 2`

output state, name: `out_state_018`, shape: `1, 256, 16, 2`

output state, name: `out_state_019`, shape: `16, 256, 11`

output state, name: `out_state_020`, shape: `16, 256, 11`

output state, name: `out_state_021`, shape: `1, 256, 8, 2`

output state, name: `out_state_022`, shape: `1, 384, 8, 2`

output state, name: `out_state_023`, shape: `1, 512, 8, 2`

output state, name: `out_state_024`, shape: `8, 256, 8`

output state, name: `out_state_025`, shape: `8, 256, 8`

output state, name: `out_state_026`, shape: `1, 512, 16, 2`

output state, name: `out_state_027`, shape: `1, 256, 16, 2`

output state, name: `out_state_028`, shape: `1, 384, 16, 2`

output state, name: `out_state_029`, shape: `1, 512, 16, 2`

output state, name: `out_state_030`, shape: `16, 128, 11`

output state, name: `out_state_031`, shape: `16, 128, 11`

output state, name: `out_state_032`, shape: `1, 256, 32, 2`

output state, name: `out_state_033`, shape: `1, 128, 32, 2`

output state, name: `out_state_034`, shape: `1, 192, 32, 2`

output state, name: `out_state_035`, shape: `1, 256, 32, 2`

output state, name: `out_state_036`, shape: `32, 64, 16`

output state, name: `out_state_037`, shape: `32, 64, 16`

output state, name: `out_state_038`, shape: `1, 128, 64, 2`

output state, name: `out_state_039`, shape: `1, 64, 64, 2`

output state, name: `out_state_040`, shape: `1, 96, 64, 2`

output state, name: `out_state_041`, shape: `1, 128, 64, 2`

output state, name: `out_state_042`, shape: `64, 32, 22`

output state, name: `out_state_043`, shape: `64, 32, 22`

output state, name: `out_state_044`, shape: `1, 64, 129, 2`

output state, name: `out_state_045`, shape: `1, 32, 129, 2`

output state, name: `out_state_046`, shape: `1, 48, 129, 2`

output state, name: `out_state_047`, shape: `1, 64, 129, 2`

output state, name: `out_state_048`, shape: `1, 129, 4`

output state, name: `out_state_049`, shape: `1, 129, 4`

output state, name: `out_state_050`, shape: `1, 129, 1`

output state, name: `out_state_051`, shape: `1, 129, 1`

## Legal Information
[*] Other names and brands may be claimed as the property of others.
