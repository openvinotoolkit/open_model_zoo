# smartlab-sequence-modelling-0001

## Use Case and High-Level Description
This is an online action segmentation network for 16 classes trained on Intel dataset. It is an online version of MSTCN++. The difference between online MSTCN++ and MSTCN++ is that the former accept stream video as input while the latter assume the whole video is given.

For the original MSTCN++ model details see [paper](https://arxiv.org/abs/2006.09220)

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| GOPs                            | 0.048915                                  |
| MParams                         | 1.018179                                  |
| Source framework                | PyTorch\*                                 |

## Accuracy
<table>
    <thead>
        <tr>
            <th colspan="2">Accuracy</th>
            <th>noise/background</th>
            <th>remove_support_sleeve</th>
            <th>adjust_rider</th>
            <th>adjust_nut</th>
            <th>adjust_balancing</th>
            <th>open_box</th>
            <th>close_box</th>
            <th>choose_weight</th>
            <th>put_left</th>
            <th>put_right</th>
            <th>take_left</th>
            <th>take_right</th>
            <th>install support_sleeve</th>
            <th>mean</th>
            <th>mPR (P+R)/2</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2>frame-level</td>
            <td rowspan=1>precision</td>
            <td>0.22</td>
            <td>0.84</td>
            <td>0.81</td>
            <td>0.62</td>
            <td>0.67</td>
            <td>0.87</td>
            <td>0.56</td>
            <td>0.52</td>
            <td>0.54</td>
            <td>0.74</td>
            <td>0.62</td>
            <td>0.68</td>
            <td>0.86</td>
            <td>0.66</td>
            <td rowspan=2>0.66</td>
        </tr>
        <tr>
            <td rowspan=1>recall</td>
            <td>0.4</td>
            <td>0.95</td>
            <td>0.83</td>
            <td>0.86</td>
            <td>0.43</td>
            <td>0.8</td>
            <td>0.31</td>
            <td>0.52</td>
            <td>0.68</td>
            <td>0.65</td>
            <td>0.62</td>
            <td>0.51</td>
            <td>0.92</td>
            <td>0.65</td>
        </tr>
        <tr>
            <td rowspan=2>segment IOU</td>
            <td rowspan=1>precision</td>
            <td>0.38</td>
            <td>0.94</td>
            <td>0.77</td>
            <td>0.65</td>
            <td>0.6</td>
            <td>0.85</td>
            <td>0.56</td>
            <td>0.68</td>
            <td>0.74</td>
            <td>0.88</td>
            <td>0.72</td>
            <td>0.78</td>
            <td>0.69</td>
            <td>0.7</td>
            <td rowspan=2>0.77</td>
        </tr>
        <tr>
            <td>recall</td>
            <td>0.64</td>
            <td>1</td>
            <td>0.96</td>
            <td>0.94</td>
            <td>0.62</td>
            <td>0.96</td>
            <td>0.48</td>
            <td>0.77</td>
            <td>0.91</td>
            <td>0.88</td>
            <td>0.83</td>
            <td>0.85</td>
            <td>1</td>
            <td>0.83</td>
        </tr>
    </tbody>
</table>

Notice: In the accuracy report, feature extraction network is i3d-rgb, you can get this model from `../../public/i3d-rgb-tf/README.md`.

## Inputs
The inputs to the network are feature vectors at each video frame, which should be the output of feature extraction network, such as i3d-rgb(`../../public/i3d-rgb-tf/README.md`) and resnet-50(`../../public/resnet-50-tf/README.md`), and feature outputs of the previous frame.

You can check the i3d-rgb and smartlab-sequence-modelling-0001 usage in demos/smartlab_demo

1. Input feature, name: `input`, shape: `1, 2048, 24`, format: `B, W, H`, where:

   - `B` - batch size
   - `W` - feature map width
   - `H` - feature map height

2. History feature 1, name: `fhis_in_0`, shape: `12, 64, 2048`, format: `C, H', W`,
3. History feature 2, name: `fhis_in_1`, shape: `11, 64, 2048`, format: `C, H', W`,
4. History feature 3, name: `fhis_in_2`, shape: `11, 64, 2048`, format: `C, H', W`,
5. History feature 4, name: `fhis_in_3`, shape: `11, 64, 2048`, format: `C, H', W`, where:

   - `C` - the channel number of feature vector
   - `H`- feature map height
   - `W` - feature map width

## Outputs

The outputs also include two parts: predictions and four feature outputs. Predictions is the action classification and prediction results. Four Feature maps are the model layer features in past frames.
1. Prediction, name: `output`, shape: `4, 1, 64, 24`, format: `C, B, H, W`,
   - `C` - the channel number of feature vector
   - `B` - batch size
   - `H`- feature map height
   - `W` - feature map width
After post-process with argmx() function, the prediction result can be used to decide the action type of the current frame.
2. History feature 1, name: `fhis_out_0`, shape: `12, 64, 2048`, format: `C, H, W`,
3. History feature 2, name: `fhis_out_1`, shape: `11, 64, 2048`, format: `C, H, W`,
4. History feature 3, name: `fhis_out_2`, shape: `11, 64, 2048`, format: `C, H, W`,
5. History feature 4, name: `fhis_out_3`, shape: `11, 64, 2048`, format: `C, H, W`, where:

   - `C` - the channel number of feature vector
   - `H`- feature map height
   - `W` - feature map width

## Legal Information
[*] Other names and brands may be claimed as the property of others.
