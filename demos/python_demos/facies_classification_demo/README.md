# Facies classification Python* Demo

This demo demonstrate how to run facies classification using OpenVINO&trade;


## How It Works
Upon the start-up, the demo application reads command-line parameters and loads a network and an given dataset file to the
Inference Engine plugin. When inference is done, the application outputs the image and displays matplotlib window with facies interpretation.

## Running

### Installation of dependencies

To install required dependencies run

```bash
pip3 install -r requirements.txt
```

Running the application with the `-h` option yields the following usage message:

``` 
python3 facies_classification_demo.py -h
```
The command yields the following usage message:
```
usage: facies_classification_demo.py [-h] -m MODEL -i DATA_PATH -l
                                     CPU_EXTENSION
                                     [-t {inline,crossline,timeline}]
                                     [-s SLICE_INDEX] [-d "<device>"]

Options:
  -h, --help            Show this help message and exit.
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model.
  -i DATA_PATH, --data_path DATA_PATH
                        Required. Path to seismic datafile.
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        Required. Required for CPU custom layers. (MaxUnpool
                        layers in this case) Absolute MKLDNN (CPU)-targeted
                        custom layers. Absolute path to a shared library with
                        the kernels implementations
  -t {inline,crossline,timeline}, --slice_type {inline,crossline,timeline}
                        Type of slice .
  -s SLICE_INDEX, --slice_index SLICE_INDEX
                        Index of slice .
  -d "<device>", --device "<device>"
                        Optional. Specify the target device to infer on: CPU,
                        GPU, FPGA, HDDL or MYRIAD. The demo will look for a
                        suitable plugin for device specified (by default, it
                        is CPU).

```

To run the demo, you can use public or pre-trained models. You can download the pre-trained models with the OpenVINO [Model Downloader](../../../tools/downloader/README.md) or from [xxx] <- (link)

## Demo Output

The application uses matplotlib to display resulting instance classification masks, saving the picture to the current directory.

![](./facies_classification_demo.png)

## See Also
* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
