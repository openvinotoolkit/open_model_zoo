# Noise Suppression Python\* Demo

This README describes the Noise Suppression demo application.

## How It Works

On startup the demo application reads command line parameters and loads a model to OpenVINO™ Runtime plugin.
It also read user-provided sound file with mix of speech and some noise to feed it into the network by small sequential patches.
The output of network is also sequence of audio patches with clean speech. The patches collected together and save into output audio file.

## Preparing to Run

The list of models supported by the demo is in `<omz_dir>/demos/noise_suppression_demo/python/models.lst` file.
This file can be used as a parameter for [Model Downloader](../../../tools/model_tools/README.md) and Converter to download and, if necessary, convert models to OpenVINO IR format (\*.xml + \*.bin).

An example of using the Model Downloader:

```sh
omz_downloader --list models.lst
```

An example of using the Model Converter:

```sh
omz_converter --list models.lst
```

### Supported Models

* noise-suppression-denseunet-ll-0001
* noise-suppression-poconetlike-0001

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.
## Running

You can use the following command to try the demo (assuming the model from the Open Model Zoo, downloaded with the
[Model Downloader](../../../tools/model_tools/README.md) executed with "--name noise-suppression*"):
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
The demo reports

* **Latency**: total processing time required to process input data (from reading the data to displaying the results).

## See Also
* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/model_tools/README.md)
* [Benchmark C++ Sample](https://docs.openvino.ai/2023.0/_inference_engine_samples_benchmark_app_README.html)
