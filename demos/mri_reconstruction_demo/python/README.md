# MRI Reconstruction Python\* Demo

This demo demonstrates MRI reconstruction model described in https://arxiv.org/abs/1810.12473 and implemented in https://github.com/rmsouza01/Hybrid-CS-Model-MRI/.
The model is used to restore undersampled MRI scans which is useful for data compression.


### Supported Models

* hybrid-cs-model-mri

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.


## Running

1. Running the application with the -h option yields the following usage message:
```bash
$ python3 mri_reconstruction_demo.py -h
usage: mri_reconstruction_demo.py [-h] -i INPUT -p PATTERN -m MODEL
                                  [-d DEVICE] [--no_show]

MRI reconstrution demo

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to input .npy file with MRI scan data.
  -p PATTERN, --pattern PATTERN
                        Path to sampling mask in .npy format.
  -m MODEL, --model MODEL
                        Path to .xml file of OpenVINO IR.
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU or
                        GPU is acceptable. Default value is
                        CPU.
  --no_show             Disable results visualization
```

2. To run the demo, you need to have
  * A sample scan from [Calgary-Campinas Public Brain MR Dataset](https://sites.google.com/view/calgary-campinas-dataset/home)
  * [Sampling mask](https://github.com/rmsouza01/Hybrid-CS-Model-MRI/blob/master/Data/sampling_mask_20perc.npy)
