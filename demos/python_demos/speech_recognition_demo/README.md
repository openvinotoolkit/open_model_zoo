# Speech Recognition Demo

This demo demonstrates Automatic Speech Recognition (ASR) with a pretrained Mozilla\* DeepSpeech 0.8.2 model.

It works with version 0.6.1 as well, and should also work with other models trained with Mozilla DeepSpeech 0.6.x/0.7.x/0.8.x with ASCII alphabets.

## How It Works

The application accepts

 * Mozilla\* DeepSpeech 0.8.2 neural network in Intermediate Representation (IR) format,
 * n-gram language model file in kenlm quantized binary format, and
 * an audio file in PCM WAV 16 kHz mono format.

After computing audio features, running a neural network to get per-frame character probabilities, and CTC decoding, the demo prints the decoded text together with the timings of the processing stages.

## Prerequisites

The demo depends on the `ctcdecode_numpy` Python extension module,
which implements CTC decoding in C++ for faster decoding.
Please refer to [Open Model Zoo demos](../../README.md) for instructions
on how to build the extension module and prepare the environment for running the demo.
Alternatively, instead of using `cmake` you can run `python -m pip install .` inside `ctcdecode-numpy` directory to build and install `ctcdecode-numpy`.

## Model preparation

You can download and convert a pre-trained Mozilla\* DeepSpeech 0.8.2 or 0.6.1 model with
OpenVINO [Model Downloader](../../../tools/downloader/README.md) and the provided conversion scripts.
This is done with the following commands:
```shell
source <openvino_dir>/bin/setupvars.sh
<openvino_dir>/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name mozilla-deepspeech-0.8.2
<openvino_dir>/deployment_tools/open_model_zoo/tools/downloader/converter.py --name mozilla-deepspeech-0.8.2
```
or
```shell
source <openvino_dir>/bin/setupvars.sh
<openvino_dir>/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name mozilla-deepspeech-0.6.1
<openvino_dir>/deployment_tools/open_model_zoo/tools/downloader/converter.py --name mozilla-deepspeech-0.6.1
```
Please pay attention to the model license, **Mozilla Public License 2.0**.

## Running Demo

Run the application with `-h` option to see help message.
Here are the available command line options:

```
usage: speech_recognition_demo.py [-h] -i FILENAME [-d DEVICE] -m FILENAME
                                  [-L FILENAME] -p NAME [-b N] [-c N]
                                  [-l FILENAME]

Speech recognition demo

optional arguments:
  -h, --help            show this help message and exit
  -i FILENAME, --input FILENAME
                        Path to an audio file in WAV PCM 16 kHz mono format
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on, for
                        example: CPU, GPU, FPGA, HDDL, MYRIAD or HETERO. The
                        sample will look for a suitable IE plugin for this
                        device. (default is CPU)
  -m FILENAME, --model FILENAME
                        Path to an .xml file with a trained model (required)
  -L FILENAME, --lm FILENAME
                        path to language model file (optional)
  -p NAME, --profile NAME
                        Choose pre/post-processing profile: mds06x_en for
                        Mozilla DeepSpeech v0.6.x, mds07x_en or mds08x_en for
                        Mozilla DeepSpeech v0.7.x/x0.8.x, other: filename of a
                        YAML file (required)
  -b N, --beam-width N  Beam width for beam search in CTC decoder (default
                        500)
  -c N, --max-candidates N
                        Show top N (or less) candidates (default 1)
  -l FILENAME, --cpu_extension FILENAME
                        Optional. Required for CPU custom layers. MKLDNN
                        (CPU)-targeted custom layers. Absolute path to a
                        shared library with the kernels implementations.
```

The typical command line is:

```shell
pip install -r requirements.txt
source <openvino_dir>/bin/setupvars.sh

python speech_recognition_demo.py \
    -p mds08x_en \
    -m <ir_dir>/FP32/mozilla_deepspeech_0.8.2.xml \
    -L <path_to_tf_model>/deepspeech-0.8.2-models.kenlm \
    -i <path_to_audio>/audio.wav
```

For version 0.6.1 it is:

```shell
python speech_recognition_demo.py \
    -p mds06x_en \
    -m <ir_dir>/FP32/mozilla_deepspeech_0.6.1.xml \
    -L <path_to_tf_model>/lm.binary \
    -i <path_to_audio>/audio.wav
```

**Only 16-bit, 16 kHz, mono-channel WAVE audio files are supported.**

An example audio file can be taken from `<openvino_dir>/deployment_tools/demo/how_are_you_doing.wav`.

## Demo Output

The application shows time taken by the initialization and processing stages, and the decoded text for the audio file.
