# Speech Recognition QuartzNet Python\* Demo

This demo demonstrates Automatic Speech Recognition (ASR) with pretrained QuartzNet model.

## How It Works

After computing audio features, running a neural network to get character probabilities, and CTC greedy decoding, the demo prints the decoded text.

## Preparing to Run

The list of models supported by the demo is in `<omz_dir>/demos/speech_recognition_quartznet_demo/python/models.lst` file.
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

* quartznet-15x5-en

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.

## Running Demo

Run the application with `-h` option to see help message.

```
usage: speech_recognition_quartznet_demo.py [-h] -m MODEL -i INPUT [-d DEVICE]

optional arguments:
  -h, --help            Show this help message and exit.
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model.
  -i INPUT, --input INPUT
                        Required. Path to an audio file in WAV PCM 16 kHz mono format
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on, for
                        example: CPU or GPU or HETERO. The
                        demo will look for a suitable OpenVINO Runtime plugin for this
                        device. Default value is CPU.
```

The typical command line is:

```sh
python3 speech_recognition_quartznet_demo.py -m quartznet-15x5-en.xml -i audio.wav
```

> **NOTE**: Only 16-bit, 16 kHz, mono-channel WAVE audio files are supported.

An example audio file can be taken from https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/librispeech_s5/how_are_you_doing_today.wav.

## Demo Output

The application prints the decoded text for the audio file.
The demo reports

* **Latency**: total processing time required to process input data (from reading the data to displaying the results).

## See Also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/model_tools/README.md)
