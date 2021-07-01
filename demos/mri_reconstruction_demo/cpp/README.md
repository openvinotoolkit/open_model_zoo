# Magnetic Resonance Image Reconstruction C++ Demo

This demo demonstrates MRI reconstruction model described in https://arxiv.org/abs/1810.12473 and implemented in https://github.com/rmsouza01/Hybrid-CS-Model-MRI/.
The model is used to restore undersampled MRI scans which is useful for data compression.


## Running

1. Running the application with the -h option yields the following usage message:
    ```bash
    $ ./mri_reconstruction_demo -h

    mri_reconstruction_demo [OPTION]
    Options:

        -h                                Print a usage message.
        -i "<path>"                       Required. Path to input .npy file with MRI scan data.
        -p "<path>"                       Required. Path to sampling mask in .npy format.
        -m "<path>"                       Required. Path to an .xml file with a trained model.
        -d "<device>"                     Optional. Specify the target device to infer on; CPU, GPU, HDDL or MYRIAD is acceptable (CPU by default).
    ```

2. To run the demo, you need to have
  * A sample scan from [Calgary-Campinas Public Brain MR Dataset](https://sites.google.com/view/calgary-campinas-dataset/home)
  * [Sampling mask](https://github.com/rmsouza01/Hybrid-CS-Model-MRI/blob/master/Data/sampling_mask_20perc.npy)
