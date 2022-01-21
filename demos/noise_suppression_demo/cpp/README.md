# Noise Suppression C++\* Demo

This README describes the Noise Suppression demo application.

## How It Works

On startup the demo application reads command line parameters and loads a network to Inference engine.
It also read user-provided sound file with mix of speech and some noise to feed it into the network by small sequential patches.
The output of network is also sequence of audio patches with clean speech. The patches collected together and save into output audio file.

## Preparing to Run

The list of models supported by the demo is in `<omz_dir>/demos/noise_suppression_demo/python/models.lst` file.
This file can be used as a parameter for [Model Downloader](../../../tools/model_tools/README.md) and Converter to download and, if necessary, convert models to OpenVINO Inference Engine format (\*.xml + \*.bin).

An example of using the Model Downloader:

```sh
omz_downloader --list models.lst
```

An example of using the Model Converter:

```sh
omz_converter --list models.lst
```

### Supported Models

* noise-suppression-poconetlike-0001
* noise-suppression-denseunet-ll-0001

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.

## Running

Running the application with the `-h` option yields the following usage message:
```sh
./noise_suppression_demo -h
```
The command yields the following usage message:
```
noise_suppression_demo [OPTION]
Options:

    -h           Print a usage message.
    -i INPUT     Required. Path to a input WAV file.
    -o OUTPUT    Optional. Path to a output WAV file.
    -m MODEL     Required. Path to an .xml file with a trained model.
    -d DEVICE    Optional. Specify the target device to infer on (the list of available devices is shown below). Default value is CPU. The demo will look for a suitable plugin for device specified.
```

Running the application with an empty list of options yields an error message.

For example, to do inference on a CPU, run the following command:

```sh
./noise_suppression_demo \
  -m <path_to_model>/noise-suppression-poconetlike-0001.xml \
  -d CPU \
  -i noisy.wav \
  -o cleaned.wav
```

## Demo Inputs

The application reads audio wave from the INPUT WAV file. The INPUT file has to have 16kHZ discretization frequency and be mono.
The MODEL is also required arguments.

## Demo Outputs
The application outputs cleaned wave to OUTPUT WAV file.
The demo reports

* **Latency**: total processing time required to process input data (from reading the data to displaying the results).

## See Also
* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvino.ai/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/model_tools/README.md)
* [Benchmark C++ Sample](https://docs.openvino.ai/latest/_inference_engine_samples_benchmark_app_README.html)
