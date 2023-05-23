# Text-to-speech Python\* Demo

The text to speech demo shows how to run the ForwardTacotron and WaveRNN models or modified ForwardTacotron and MelGAN models to produce an audio file for a given input text file.
The demo is based on https://github.com/seungwonpark/melgan, https://github.com/as-ideas/ForwardTacotron and https://github.com/fatchord/WaveRNN repositories.

## How It Works

On startup, the demo application reads command-line parameters and loads four or three models to OpenVINO™ Runtime plugin. The demo pipeline reads text file by lines and divides every line to parts by punctuation marks.
The heuristic algorithm chooses punctuation near to the some threshold by sentence length.
When inference is done, the application outputs the audio to the WAV file with 22050 Hz sample rate.

## Preparing to Run

The list of models supported by the demo is in `<omz_dir>/demos/text_to_speech_demo/python/models.lst` file.
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

* forward-tacotron-duration-prediction
* forward-tacotron-regression
* wavernn-rnn
* wavernn-upsampler
* text-to-speech-en-0001-duration-prediction
* text-to-speech-en-0001-generation
* text-to-speech-en-0001-regression
* text-to-speech-en-multi-0001-duration-prediction
* text-to-speech-en-multi-0001-generation
* text-to-speech-en-multi-0001-regression

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.

## Running

Running the application with the `-h` option yields the following usage message:

```
usage: text_to_speech_demo.py [-h] -m_duration MODEL_DURATION -m_forward
                              MODEL_FORWARD -i INPUT [-o OUT] [-d DEVICE]
                              [-m_upsample MODEL_UPSAMPLE] [-m_rnn MODEL_RNN]
                              [--upsampler_width UPSAMPLER_WIDTH]
                              [-m_melgan MODEL_MELGAN] [-s_id SPEAKER_ID]
                              [-a ALPHA]

Options:
  -h, --help            Show this help message and exit.
  -m_duration MODEL_DURATION, --model_duration MODEL_DURATION
                        Required. Path to ForwardTacotron`s duration
                        prediction part (*.xml format).
  -m_forward MODEL_FORWARD, --model_forward MODEL_FORWARD
                        Required. Path to ForwardTacotron`s mel-spectrogram
                        regression part (*.xml format).
  -i INPUT, --input INPUT
                        Required. Text or path to the input file.
  -o OUT, --out OUT     Optional. Path to an output .wav file
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU,
                        GPU or HETERO is acceptable. The
                        demo will look for a suitable plugin for device
                        specified. Default value is CPU
  -m_upsample MODEL_UPSAMPLE, --model_upsample MODEL_UPSAMPLE
                        Path to WaveRNN`s part for mel-spectrogram upsampling
                        by time axis (*.xml format).
  -m_rnn MODEL_RNN, --model_rnn MODEL_RNN
                        Path to WaveRNN`s part for waveform autoregression
                        (*.xml format).
  --upsampler_width UPSAMPLER_WIDTH
                        Width for reshaping of the model_upsample in WaveRNN
                        vocoder. If -1 then no reshape. Do not use with FP16
                        model.
  -m_melgan MODEL_MELGAN, --model_melgan MODEL_MELGAN
                        Path to model of the MelGAN (*.xml format).
  -s_id SPEAKER_ID, --speaker_id SPEAKER_ID
                        Ordinal number of the speaker in embeddings array for
                        multi-speaker model. If -1 then activates the multi-
                        speaker TTS model parameters selection window.
  -a ALPHA, --alpha ALPHA
                        Coefficient for controlling of the speech time
                        (inversely proportional to speed).
```

Running the application with the empty list of options yields the usage message and an error message.

## Example for Running with Arguments

### Speech synthesis with ForwardTacotron and WaveRNN models

```sh
python3 text_to_speech_demo.py \
    --input <path_to_file>/text.txt \
    -o <path_to_audio>/audio.wav \
    --model_duration <path_to_model>/forward_tacotron_duration_prediction.xml \
    --model_forward <path_to_model>/forward_tacotron_regression.xml \
    --model_upsample <path_to_model>/wavernn_upsampler.xml \
    --model_rnn <path_to_model>/wavernn_rnn.xml
```

> **NOTE**: You can use `--upsampler_width` parameter for this demo for the purpose of control width of the time axis
> in the input mel-spectrogram for the `wavernn_upsampler` network. This option can help you improve the speed of
> the pipeline inference on the long sentences.

### Speech synthesis with text-to-speech-en-0001 models

```sh
python3 text_to_speech_demo.py \
    -i <path_to_file>/text.txt \
    -o <path_to_audio>/audio.wav \
    -m_duration <path_to_model>/text-to-speech-en-0001-duration-prediction.xml \
    -m_forward <path_to_model>/text-to-speech-en-0001-regression.xml \
    -m_melgan <path_to_model>/text-to-speech-en-0001-generation.xml
```

### Speech synthesis with multi-speaker text-to-speech-en-multi-0001 models

```sh
python3 text_to_speech_demo.py \
    -i <path_to_file>/text.txt \
    -o <path_to_audio>/audio.wav \
    -s_id 19 \
    -m_duration <path_to_model>/text-to-speech-en-multi-0001-duration-prediction.xml \
    -m_forward <path_to_model>/text-to-speech-en-multi-0001-regression.xml \
    -m_melgan <path_to_model>/text-to-speech-en-multi-0001-generation.xml
```

> **NOTE**: `s_id` defines the style of the speaker utterance. You can choose it equal to -1 to activate the
> multi-speaker TTS model parameters selection window. This window provides an opportunity to choose the gender of the
> speaker, index number of the speaker or calculate PCA based speaker embedding. The `s_id` is available only for
> `text-to-speech-en-multi-0001` models.


## Demo Output

The application outputs WAV file with generated audio.
The demo reports

* **Latency**: total processing time required to process input data (from reading the data to displaying the results).

## See Also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/model_tools/README.md)
