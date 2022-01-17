# online mstcn++

## Use Case and High-Level Description
This is an online action segmentation network for 16 classes trained on Intel dataset. It is an online version of MSTCN++. The difference between online MSTCN++ and MSTCN++ is that the former accept stream video as input while the latter assume the whole video is given.

For the original MSTCN++ model details see [paper](https://arxiv.org/abs/2006.09220)

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| TODO Accuracy                   | TODO                                          |
| GOPs                            | 0.048915                                  |
| MParams                         | 1.018179                                  |
| Source framework                | PyTorch\*                                 |

## Inputs
The inputs to the network are feature vectors at each video frame, which should be the output of feature extraction network, such as i3d-rgb and resnet-50, and feature outputs of the previous frame.

1. Input feature, name: `input`, shape: `1, 2048, 24`, format: `B, W, H`, where:

   - `B` - batch size
   - `W` - feature map width
   - `H` - feature map height

2. History feature 1, name: `fhis_in_0`, shape: `12, 64, 2048`, format: `C, H', W`,
3. History feature 2, name: `fhis_in_1`, shape: `11, 64, 2048`, format: `C, H', W`,
4. History feature 3, name: `fhis_in_2`, shape: `11, 64, 2048`, format: `C, H', W`,
5. History feature 4, name: `fhis_in_3`, shape: `11, 64, 2048`, format: `C, H', W`, where:

   - `C` - the channel number of feature vector
   - `H'`- feature map height
   - `W` - feature map width

## Outputs

The outputs also include two parts: predictions and four feature outputs. Predictions is the action classification and prediction results. Four Feature maps are the model layer features in past frames.
1. Prediction, name: `output`, shape: `4, 1, 64, 24`, format: `C, B, H', W`,
   - `C` - the channel number of feature vector
   - `B` - batch size
   - `H'`- feature map height
   - `W` - feature map width
After post-process with argmx() function, the prediction result can be used to decide the action type of the current frame.
2. History feature 1, name: `fhis_out_0`, shape: `12, 64, 2048`, format: `C, H', W`,
3. History feature 2, name: `fhis_out_1`, shape: `11, 64, 2048`, format: `C, H', W`,
4. History feature 3, name: `fhis_out_2`, shape: `11, 64, 2048`, format: `C, H', W`,
5. History feature 4, name: `fhis_out_3`, shape: `11, 64, 2048`, format: `C, H', W`, where:

   - `C` - the channel number of feature vector
   - `H'`- feature map height
   - `W` - feature map width

## Legal Information

The original model is distributed under the following
[license](https://github.com/fatchord/WaveRNN/blob/master/LICENSE.txt)

```
MIT License

Copyright (c) 2019 fatchord (https://github.com/fatchord)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
