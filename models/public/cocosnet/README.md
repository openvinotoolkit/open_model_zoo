# cocosnet

## Use Case and High-Level Description

Cross-domain correspondence network is a exemplar-based image translation model, consisting of correspondence and translation parts. Model was pre-trained on ADE20k dataset.
For details see [paper](https://arxiv.org/abs/2004.05571) and [repository](https://github.com/microsoft/CoCosNet).

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Image translation                         |
| GFLOPs                          | 1080.7032                                 |
| MParams                         | 167.9141                                  |
| Source framework                | PyTorch\*                                 |

## Accuracy

Metrics were calculated between generated images by model and real validation images from ADE20k dataset.
For some GAN metrics (IS and FID) you need to use classification model as verification network.
In our case it is [Inception-V3](../googlenet-v3/README.md) model.
For details, please check Accuracy Checker [config](accuracy-check-pipelined.yml).

| Metric | Original model | Converted model |
| ------ | -------------- | --------------- |
| PSNR   | 12.99dB        | 12.93dB         |
| SSIM   | 0.34           | 0.34            |
| IS     | 13.34          | 13.35           |
| FID    | 33.27          | 33.14           |

### Inputs

1. name: `input_seg_map`, shape: `1, 151, 256, 256` - Input semantic segmentation mask (one-hot label map) in the format `B, C, H, W`, where:

    - `B` - batch size
    - `C` - number of classes (151 for ADE20k)
    - `H` - mask height
    - `W` - mask width

2. name: `ref_image`, shape: `1, 3, 256, 256` - An reference image (exemplar) in the format `B, C, H, W`, where:

    - `B` - batch size
    - `C` - number of channels
    - `H` - image height
    - `W` - image width

    Expected color order is `BGR` (At original model expected color order is `RGB`).

3. name: `ref_seg_map`, shape: `1, 151, 256, 256` - A mask (one-hot label map) for reference image in the format `B, C, H, W`, where:

    - `B` - batch size
    - `C` - number of classes (151 for ADE20k)
    - `H` - mask height
    - `W` - mask width

### Output

Image, name: `exemplar_based_output`, shape: `1, 3, 256, 256` - A result (generated) image based on exemplar in the format `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Output color order is `RGB`.

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

The original model is distributed under the following
[license](https://github.com/microsoft/CoCosNet/blob/master/LICENSE):

```
MIT License

Copyright (c) Microsoft Corporation.

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
SOFTWARE
```

The Synchronized-BatchNorm-PyTorch (dependency for model) is distributed under the following
[license](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch/blob/master/LICENSE):

'''
MIT License

Copyright (c) 2018 Jiayuan MAO

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
'''
