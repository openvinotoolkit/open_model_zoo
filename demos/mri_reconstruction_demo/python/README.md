# Magnetic Resonance Image Reconstruction Python Demo

This demo demonstrates MRI reconstruction model described in https://arxiv.org/abs/1810.12473 and implemented in https://github.com/rmsouza01/Hybrid-CS-Model-MRI/.
The model is used to restore undersampled MRI scans which is useful for data compression.


## Running

1. Running the application with the -h option yields the following usage message:
    ```bash
    $ python3 mri_reconstruction_demo.py -h
    usage: mri_reconstruction_demo.py [-h] [-i INPUT] [-p PATTERN] [-m MODEL]
                                      [-d DEVICE]

    MRI reconstrution demo for network from https://github.com/rmsouza01/Hybrid-
    CS-Model-MRI (https://arxiv.org/abs/1810.12473)

    optional arguments:
      -h, --help            show this help message and exit
      -i INPUT, --input INPUT
                            Path to input .npy file with MRI scan data.
      -p PATTERN, --pattern PATTERN
                            Path to sampling mask in .npy format.
      -m MODEL, --model MODEL
                            Path to .xml file of OpenVINO IR.
      -d DEVICE, --device DEVICE
                            Optional. Specify the target device to infer on; CPU,
                            GPU, HDDL or MYRIAD is acceptable. For non-CPU
                            targets, HETERO plugin is used with CPU fallbacks to
                            FFT implementation. Default value is CPU
    ```

3. To run the demo, you need to have
  * A sample scan from [Calgary-Campinas Public Brain MR Dataset](https://sites.google.com/view/calgary-campinas-dataset/home)
  * [Sampling mask](https://github.com/rmsouza01/Hybrid-CS-Model-MRI/blob/master/Data/sampling_mask_20perc.npy)
