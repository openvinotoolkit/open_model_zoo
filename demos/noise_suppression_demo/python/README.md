# Noise Suppression Python\* Demo

This README describes the Noise Suppresion demo application.

## How It Works

On startup the demo application reads command line parameters and loads a network to Inference engine.
It also read user-provided sound file with mix of speech and some noise to feed it into the network by small sequential patches.
The output of network is also sequence of audio patches with clean speech. The patches collected together and save into ouput audio file.

## Preparing to Run

The list of models supported by the demo is in <omz_dir>/demos/noise_suppression_demo/python/models.lst file.
This file can be used as a parameter for [Model Downloader](../../../tools/downloader/README.md) and Converter to download and, if necessary, convert models to OpenVINO Inference Engine format (\*.xml + \*.bin).

An example of using the Model Downloader:

```sh
python3 <omz_dir>/tools/downloader/downloader.py --list models.lst
```

An example of using the Model Converter:

```sh
python3 <omz_dir>/tools/downloader/converter.py --list models.lst
```

### Supported Models

* noise-suppression-poconetlike-0001

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.
## Running

Running the application with the `-h` option yields the following usage message:
```
python3 noise_suppression_demo.py -h
```
The command yields the following usage message:
```
usage: noise_suppression_demo.py [-h] -m MODEL -i INPUT [-o OUTPUT] [-d DEVICE]

Options:
  -h, --help            Show this help message and exit.
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model
  -i INPUT, --input INPUT
                        Required. Path to a 16kHz wav file with speech+noise
  -o OUTPUT, --output OUTPUT
                        Optional. Path to output wav file for cleaned speech
  -d DEVICE, --device DEVICE
                        Optional. Target device to perform inference on.
                        Default value is CPU
```

You can use the following command to try the demo (assuming the model from the Open Model Zoo, downloaded with the
[Model Downloader](../../../tools/downloader/README.md) executed with "--name noise-suppression*"):
```
    python3 noise_suppression_demo.py \
        --model=<path_to_model>/noise-suppression-poconetlike-0001.xml \
        --input=noisy.wav \
        --output=cleaned.wav
```

## Demo Inputs

The application reads audio wave from the input file with given name. The input file has to have 16kHZ discretization frequency
The model is also required demo arguments.

## Demo Outputs
The application outputs cleaned wave to output file.

## Demo Performance
Even though the demo reports inference performance (by measuring wall-clock time for individual inference calls),
it is only baseline performance.
Please use the full-blown [Benchmark C++ Sample](https://docs.openvinotoolkit.org/latest/_inference_engine_samples_benchmark_app_README.html)
for any actual performance measurements.


## See Also
* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
* [Benchmark C++ Sample](https://docs.openvinotoolkit.org/latest/_inference_engine_samples_benchmark_app_README.html)
