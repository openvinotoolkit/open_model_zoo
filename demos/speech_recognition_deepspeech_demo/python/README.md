# Speech Recognition DeepSpeech Python\* Demo

This demo shows Automatic Speech Recognition (ASR) with a pretrained Mozilla\* DeepSpeech 0.8.2 model.

It works with version 0.6.1 as well, and should also work with other models trained with Mozilla DeepSpeech 0.6.x/0.7.x/0.8.x/0.9.x with ASCII alphabets.

## How It Works

The application accepts

* Mozilla\* DeepSpeech 0.8.2 neural network in Intermediate Representation (IR) format,
* n-gram language model file in kenlm quantized binary format, and
* an audio file in PCM WAV 16 kHz mono format.

The application has two modes:

 * *Normal mode* (default). Audio data is streamed in 10 second chunks into a streaming pipeline of: computation of audio features, running a neural network to get per-frame character probabilities, and CTC decoding. After processing the whole file, the demo prints the decoded text and the time spent.

 * In *simulated real-time mode* the app simulates speech recognition of live recording by feeding audio data from input file and displaying the current partial result in a creeping line in console output. Data is fed at real-time speed by introducing the necessary delays. Audio data is fed in 0.32 sec chunks (size is controlled by `--block-size` option) into the same streaming pipeline. In this mode the pipeline provides updated recognition result after each data chunk.

## Preparing to Run

The list of models supported by the demo is in `<omz_dir>/demos/speech_recognition_deepspeech_demo/python/models.lst` file.
This file can be used as a parameter for [Model Downloader](../../../tools/model_tools/README.md) and Converter to download and, if necessary, convert models to OpenVINO IR format (\*.xml + \*.bin).
Don't forget to configure Model Optimizer, which is a requirement for Model Downloader, as described in its documentation.

An example of using the Model Downloader:

```sh
omz_downloader --list models.lst
```

An example of using the Model Converter:

```sh
omz_converter --list models.lst
```

Please pay attention to the model license, **Mozilla Public License 2.0**.

## Prerequisites

The demo depends on the `ctcdecode_numpy` Python extension module, which implements CTC decoding in C++ for faster decoding.
Please refer to [Open Model Zoo demos](../../README.md) for instructions
on how to build the extension module and prepare the environment for running the demo.
Alternatively, instead of using `cmake` you can run `python -m pip install .` inside `ctcdecode-numpy` directory to build and install `ctcdecode-numpy`.

### Supported Models

* mozilla-deepspeech-0.6.1
* mozilla-deepspeech-0.8.2

Please pay attention to the model license, **Mozilla Public License 2.0**.

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.

## Running Demo

Run the application with `-h` option to see help message.
Here are the available command line options:

```
usage: speech_recognition_deepspeech_demo.py [-h] -i FILENAME [-d DEVICE] -m
                                             FILENAME [-L FILENAME] -p NAME
                                             [-b N] [-c N] [--realtime]
                                             [--block-size BLOCK_SIZE]
                                             [--realtime-window REALTIME_WINDOW]

Speech recognition DeepSpeech demo

optional arguments:
  -h, --help            show this help message and exit
  -i FILENAME, --input FILENAME
                        Required. Path to an audio file in WAV PCM 16 kHz mono format
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on, for
                        example: CPU or GPU or HETERO. The
                        demo will look for a suitable OpenVINO Runtime plugin for this
                        device. (default is CPU)
  -m FILENAME, --model FILENAME
                        Required. Path to an .xml file with a trained model
  -L FILENAME, --lm FILENAME
                        Optional. Path to language model file
  -p NAME, --profile NAME
                        Required. Choose pre/post-processing profile: mds06x_en
                        for Mozilla DeepSpeech v0.6.x,
                        mds07x_en/mds08x_en/mds09x_en for Mozilla DeepSpeech
                        v0.7.x/v0.8.x/v0.9.x(English), other: filename of a
                        YAML file
  -b N, --beam-width N  Beam width for beam search in CTC decoder (default
                        500)
  -c N, --max-candidates N
                        Show top N (or less) candidates (default 1)
  --realtime            Simulated real-time mode: slow down data feeding to
                        real time and show partial transcription during
                        recognition
  --block-size BLOCK_SIZE
                        Block size in audio samples for streaming into ASR
                        pipeline (defaults to samples in 10 sec for offline;
                        samples in 16 frame strides for online)
  --realtime-window REALTIME_WINDOW
                        In simulated real-time mode, show this many characters
                        on screen (default 79)
```

The typical command line for offline mode is:

```shell
python3 speech_recognition_deepspeech_demo.py \
    -p mds08x_en \
    -m <path_to_model>/mozilla-deepspeech-0.8.2.xml \
    -L <path_to_file>/deepspeech-0.8.2-models.kenlm \
    -i <path_to_audio>/audio.wav
```

For version 0.6.1 it is:

```shell
python3 speech_recognition_deepspeech_demo.py \
    -p mds06x_en \
    -m <path_to_model>/mozilla-deepspeech0-0.6.1.xml \
    -L <path_to_file>/lm.binary \
    -i <path_to_audio>/audio.wav
```

To run in *simulated real-time mode* add command-line option `--realtime`.

> **NOTE**: Only 16-bit, 16 kHz, mono-channel WAVE audio files are supported.

Optional (but highly recommended) language model files, `deepspeech-0.8.2-models.kenlm` or `lm.binary` are part of corresponding model downloaded content and will be located in the Model Downloader output folder after model downloading and conversion.
An example audio file can be taken from https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/librispeech_s5/how_are_you_doing_today.wav.

## Demo Output

The application shows time taken by the initialization and processing stages, and the decoded text for the audio file. In real-time mode the current recognition result is shown while the app is running as well.
In offline mode the demo reports

* **Latency**: total processing time required to process input data (from reading the data to displaying the results).

## See Also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/model_tools/README.md)
