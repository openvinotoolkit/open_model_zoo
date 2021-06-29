# Speech Recognition QuartzNet Python\* Demo

This demo demonstrates Automatic Speech Recognition (ASR) with pretrained QuartzNet model.

## How It Works

After computing audio features, running a neural network to get character probabilities, and CTC greedy decoding, the demo prints the decoded text.

## Preparing to Run

The list of models supported by the demo is in `<omz_dir>/demos/speech_recognition_quartznet_demo/python/models.lst` file.
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
                        Path to an audio file in WAV PCM 16 kHz mono format
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on, for
                        example: CPU, GPU, HDDL, MYRIAD or HETERO. The
                        demo will look for a suitable IE plugin for this
                        device. Default value is CPU.
```

The typical command line is:

```sh
python3 speech_recognition_quartznet_demo.py -m quartznet-15x5-en.xml -i audio.wav
```

> **NOTE**: Only 16-bit, 16 kHz, mono-channel WAVE audio files are supported.

An example audio file can be taken from `<openvino_dir>/deployment_tools/demo/how_are_you_doing.wav`.

## Demo Output

The application prints the decoded text for the audio file.

## See Also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
