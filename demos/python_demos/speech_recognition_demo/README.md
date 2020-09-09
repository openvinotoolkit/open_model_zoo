# Speech Recognition Demo

This demo demonstrates Automatic Speech Recognition (ASR) with a pretrained Mozilla\* DeepSpeech 0.6.1 model.

## How It Works

The application accepts

 * Mozilla\* DeepSpeech 0.6.1 neural network in Intermediate Representation (IR) format,
 * n-gram language model file in kenlm quantized binary format, and
 * an audio file in PCM WAV 16 kHz mono format.

After computing audio features, running a neural network to get per-frame character probabilities, and CTC decoding, the demo prints the decoded text together with the timings of the processing stages.

The app depends on `ctcdecode_numpy` Python\* module, its installation is described below.

## Model preparation

You can download and convert a pre-trained Mozilla\* DeepSpeech 0.6.1 model with
OpenVINO [Model Downloader](../../../tools/downloader/README.md).
This essentially boils down to the following commands:
```shell
source <openvino_path>/bin/setupvars.sh
<openvino_path>/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name mozilla-deepspeech-0.6.1
<openvino_path>/deployment_tools/open_model_zoo/tools/downloader/converter.py --name mozilla-deepspeech-0.6.1
```

Please pay attention to the model license, **Mozilla Public License 2.0**.


## Installing CTC decoder module

ASR performance depends heavily on beam width (a.k.a. beam size), which is the number of candidate strings maintained by beam search on each iteration.
Using larger beam results in better recognition, but is slower.
The demo depends on `ctcdecode_numpy` Python module: it implements CTC decoding in C++ for faster decoding.

To install `ctcdecode_numpy` Python module either follow ["Build the Native Python* Extension Modules"](../../README.md#build_python_extensions),
or install it with pip:

1. Create and activate a virtualenv, it you haven't already:

    ```shell
    virtualenv -p python3 --system-site-packages deepspeech-venv
    . deepspeech-venv/bin/activate
    ```

1. Build and install `ctcdecode_numpy` Python module:

    ```shell
    cd ctcdecode-numpy/
    python -m pip install .
    ```


## Running Demo

Run the application with `-h` option to see help message.
Here are the essential options:

```
usage: speech_recognition_demo.py [-h] -i FILENAME [-d DEVICE] -m FILENAME
                                  [-b N] [-L FILENAME] [-a FILENAME]
                                  [--alpha X] [--beta X] [-l FILENAME]

Speech recognition example

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
  -b N, --beam-width N  Beam width for beam search in CTC decoder (default
                        500)
  -L FILENAME, --lm FILENAME
                        path to language model file (optional)
[...]
```

The typical command line is:

```shell
pip install -r requirements.txt
source <openvino_path>/bin/setupvars.sh

python speech_recognition_demo.py \
    -m <ir_dir>/mozilla_deepspeech_0.6.1.xml \
    -L <path_to_tf_model>/lm.binary \
    <path_to_audio>/audio.wav
```

**Only 16-bit, 16 kHz, mono-channel WAVE audio files are supported.**

An example audio file can be taken from `<openvino directory>/deployment_tools/demo/how_are_you_doing.wav`.

## Demo Output

The application shows the time taken by initialization and processing stages, and the decoded text for the audio file.
