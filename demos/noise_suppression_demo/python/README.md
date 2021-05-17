# Noise Suppression Python\* Demo

This README describes the Noise Suppresion demo application.

## How It Works

Upon the start-up the demo application reads command line parameters and loads a network to Inference engine.
It also read user-provided sound file with mix of speech and some noise to feed it into the network by small sequential patches.
The output of network is also sequence of audio patches with clean speech. The patches collected together and save into ouput audio file.

## Preparing to Run

The list of models supported by the demo is in <omz_dir>/demos/noise_suppression_demo/python/models.lst file.
This file can be used as a parameter for [Model Downloader](../../../tools/downloader/README.md) and Converter to download and, if necessary, convert models to OpenVINO Inference Engine format (\*.xml + \*.bin).

### Supported Models

* noise-suppression-poconetlike-0001

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.
## Running

Running the application with the `-h` option yields the following usage message:
```
python3 noise_suppression.py -h
```
The command yields the following usage message:
```
usage: noise_suppression.py [-h] -m MODEL -i INPUT -o OUTPUT

Options:
  -h, --help            Show this help message and exit.
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model
  -i INPUT, --input INPUT
                        Required. path to sound file with speech and noise mix
  -o OUTPUT, --output OUTPUT
                        Required. path to sound file with clean speech
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU
                        is acceptable. Sample will look for a suitable plugin
                        for device specified. Default value is CPU
```

> **NOTE**: Before running the demo with a trained model, make sure to convert the model to the Inference Engine's
> Intermediate Representation format (\*.xml + \*.bin)
> using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).
> When using the pre-trained model from the model zoo (please see [Model Downloader](../../../tools/downloader/README.md)),
> the model is already converted to the IR.

## Demo Inputs

The application reads audio wave from the input file with given name. The input file has to have 16kHZ discretization frequency
The model is also important demo arguments.

## Demo Outputs
The application outputs cleaned wave to output file.

## Supported Models
[Open Model Zoo Models](../../../models/intel/index.md) feature
example noise-suppression-*.

[OpenVINO Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).
Specifically the example command-line is as follows:
```
    python3 mo.py -m <path_to_model>/noise-suppression-poconetlike-0001.onnx
```

## Example Demo Cmd-Line
You can use the following command to try the demo (assuming the model from the Open Model Zoo, downloaded with the
[Model Downloader](../../../tools/downloader/README.md) executed with "--name dns*"):
```
    python3 noise_suppression.py
            --model=<path_to_model>/noise-suppression-poconetlike-0001.xml
            --input=noisy.wav
            --output=cleaned.wav
```

## Demo Performance
Even though the demo reports inference performance (by measuring wall-clock time for individual inference calls),
it is only baseline performance.
Please use the full-blown [Benchmark C++ Sample](https://docs.openvinotoolkit.org/latest/_inference_engine_samples_benchmark_app_README.html)
for any actual performance measurements.


## See Also
* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
* [Benchmark C++ Sample](https://docs.openvinotoolkit.org/latest/_inference_engine_samples_benchmark_app_README.html)
