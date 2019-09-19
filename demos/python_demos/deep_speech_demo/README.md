Deep Speech Python* Demo
===============================

This is the demo application for Deep Speech algorithm, which make speech to text that are being performed on input speech audio.
Following the below command download the pre-trained models:
 - For UNIX*-like systems, run the following command:
    ```sh
    wget -O - https://github.com/mozilla/DeepSpeech/releases/download/v0.5.0/deepspeech-0.5.0-models.tar.gz | tar xvfz -
    ```
 - For Windows*-like systems:
    1. Download the archive from the DeepSpeech project repository: https://github.com/mozilla/DeepSpeech/releases/download/v0.5.0/deepspeech-0.5.0-models.tar.gz.
    2. Unzip it twice with a file archiver application.

To generate the DeepSpeech Intermediate Representation (IR), provide TensorFlow DeepSpeech model to the Model Optimizer with parameters:

```sh
python3 ./mo_tf.py
--input_model path_to_model/output_graph.pb                         \
--freeze_placeholder_with_value input_lengths->[16]                 \
--input input_node,previous_state_h/read,previous_state_c/read      \
--input_shape [1,16,19,26],[1,2048],[1,2048]                        \
--output raw_logits,lstm_fused_cell/Gather,lstm_fused_cell/Gather_1 \ 
--disable_nhwc_to_nchw
```

For more information about this, refer to the [Convert Tensorflow* DeepSpeech Model to the Intermediate Representation](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_DeepSpeech_From_Tensorflow.html).

Running
-------
Running the application with the `-h` option yields the following usage message:

```
usage: deepspeech_openvino_0.5.py [-h] -m MODEL -i AUDIO -a ALPHABET
                                  [-l CPU_EXTENSION] 
                                  [-d DEVICE]

Options:
  -h, --help            show this help message and exit
  -m  MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model.
  -i AUDIO, --input AUDIO
                        Required. Required. Path to an audio files.
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        Optional. For CPU custom layers, if any. Absolute path
                        to a shared library with the kernels implementation.
  -d DEVICE, --device DEVICE
                        Optional. Specify a target device to infer on. CPU, GPU, FPGA, HDDL or MYRIAD is
                        acceptable. The demo will look for a suitable plugin for the device specified.
                        Default value is CPU
  -a ALPHABET, --alphabet ALPHABET
                        Required. Path to a alphabet file.
```

Running Demo

```sh
python3 deepspeech_openvino_0.5.py -m models/deepspeech_0.5.0.xml \
    -i <path_to_audio>/audio.wav \
    -a alphabet_b.txt
```
Only 16-bit, 16 kHz, mono-channel WAVE audio files are supported. One example wave file can be downloaded from https://github.com/jcsilva/docker-kaldi-gstreamer-server/raw/master/audio/1272-128104-0000.wav.

Demo Output
------------
The application shows the text output for speech audio.
