# Speech Recognition Offline Python\* Demo

This demo demonstrates Automatic Speech Recognition (ASR) with pretrained QuartzNet model.

## How It Works

After computing audio features, running a neural network to get character probabilities, and CTC greedy decoding, the demo prints the decoded text.

## Preparing to run

Pre-trained models, supported by demo listed in [models.lst](./models.lst) file, located at each demo folder.
This file can be used as a parameter for [Model Downloader](../../../tools/downloader/README.md) and Converter to download and, if necessary, convert models to OpenVINO Inference Engine format (\*.xml + \*.bin).

### Supported models

* quartznet-15x5-en

> **NOTE**: Refer to tables for [Intel](../../../models/intel/device_support.md) and [public](../../../models/public/device_support.md) models which summarize models support at different devices to select target inference device.

## Running Demo

Run the application with `-h` option to see help message.

```
usage: speech_recognition_offline_demo.py [-h] -m MODEL -i INPUT [-d DEVICE]

optional arguments:
  -h, --help            Show this help message and exit.
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model.
  -i INPUT, --input INPUT
                        Path to an audio file in WAV PCM 16 kHz mono format
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on, for
                        example: CPU, GPU, FPGA, HDDL, MYRIAD or HETERO. The
                        sample will look for a suitable IE plugin for this
                        device. Default value is CPU.
```

The typical command line is:

```sh
python3 speech_recognition_offline_demo.py -m quartznet-15x5-en.xml -i audio.wav
```

**Only 16-bit, 16 kHz, mono-channel WAVE audio files are supported.**

An example audio file can be taken from `<openvino_dir>/deployment_tools/demo/how_are_you_doing.wav`.

## Demo Output

The application prints the decoded text for the audio file.

## See Also

* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
