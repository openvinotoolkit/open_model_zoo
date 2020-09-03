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

1. Download and unzip pre-trained models from Mozilla DeepSpeech repository:
https://github.com/mozilla/DeepSpeech/releases/download/v0.6.1/deepspeech-0.6.1-models.tar.gz

    Please pay attention to the model license, Mozilla Public License 2.0.

2. Generate OpenVINO Inference Engine Intermediate Representation (IR) from the downloaded model using OpenVINO model optimizer (MO)

    ```shell
    python <path_to_mo>/mo_tf.py --model_name "deepspeech_0.6.1" \
      --output_dir <output_ir_dir> \
      --input_model <path_to_tf_model>/output_graph.pb \
      --freeze_placeholder_with_value "input_lengths->[16]" \
      --input "input_node,previous_state_h,previous_state_c" \
      --input_shape "[1,16,19,26],[1,2048],[1,2048]" \
      --output "logits,cudnn_lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/GatherNd,cudnn_lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/GatherNd_1" \
      --disable_nhwc_to_nchw \
      --log_level WARNING
    ```

## Installing CTC decoder module

ASR performance depends heavily on beam width (a.k.a. beam size), which is the number of candidate strings maintained by beam search on each iteration.
Using larger beam results in better recognition, but is slower.

The demo depends on `ctcdecode_numpy` Python module: it implements CTC decoding in C++ for faster decoding.

To install `ctcdecode_numpy` Python module:

1. Create and activate a virtualenv, it you haven't already:

    ```shell
    virtualenv -p python3 --system-site-packages deepspeech-venv
    . deepspeech-venv/bin/activate
    ```

1. Install `swig` utility.  For Ubuntu 18.04 or 16.04, use:

    ```shell
    sudo apt install swig
    ```

    For Windows [download](http://www.swig.org/download.html) and unzip `swigwin-*.zip` file, and add the directory containing `swig.exe` to your PATH environment valiable.

1. Build and install `ctcdecode_numpy` Python module:

    ```shell
    cd ctcdecode-numpy/
    pip install -r requirements.txt
    python setup.py build_ext install
    ```

1. Check that loading `ctcdecode_numpy` doesn't produce any errors:

    ```shell
    cd ..
    python -c "import ctcdecode_numpy"
    ```

    If you encounter any problem, don't forget to run `clean.sh` before compiling again.


## Running Demo

Run the application with `-h` option to see help message.
Here are the essential options:

```
usage: speech_recognition_demo.py [-h] [-d DEVICE] -m FILENAME [-b N]
                                  [-L FILENAME] [-a FILENAME] [--alpha X]
                                  [--beta X] [-l FILENAME]
                                  FILENAME

Speech recognition example

positional arguments:
  FILENAME              Path to an audio file. Must be WAV PCM 16 kHz mono
                        format.

optional arguments:
  -h, --help            show this help message and exit
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU,
                        GPU, FPGA, HDDL, MYRIAD or HETERO: is acceptable. The
                        sample will look for a suitable plugin for device
                        specified. Default value is CPU
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

python deepspeech_0.6.1_openvino_demo.py \
    -m <ir_dir>/mozilla_deepspeech_0.6.1.xml \
    -L <path_to_tf_model>/lm.binary \
    <path_to_audio>/audio.wav
```

**Only 16-bit, 16 kHz, mono-channel WAVE audio files are supported.**

An example audio file can be taken from `<openvino directory>/deployment_tools/demo/how_are_you_doing.wav`.

## Demo Output

The application shows the time taken by initialization and processing stages, and the decoded text for the audio file.
