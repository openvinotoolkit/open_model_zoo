# Sound Classification Python\* Demo

Demo application for sound classification algorithm.

## How It Works

On startup the demo application reads command line parameters and loads a model to OpenVINO™ Runtime plugin. It uses only audio files in `wav` format. Audio should be converted to model's sample rate using `-sr/--sample_rate` option, if sample rate of audio differs from sample rate of model (e.g. [AclNet](../../../models/public/aclnet/README.md) expected 16kHz audio). After reading the audio, it is sliced into clips to fit model input (clips are allowed to overlap with `-ol/--overlap` option) and each clip is processed separately with its own prediction.

## Preparing to Run

For demo input image or video files, refer to the section **Media Files Available for Demos** in the [Open Model Zoo Demos Overview](../../README.md).
The list of models supported by the demo is in `<omz_dir>/demos/sound_classification_demo/python/models.lst` file.
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

* aclnet
* aclnet-int8

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.

## Running

Run the application with the `-h` option to see the usage message:

```
usage: sound_classification_demo.py [-h] -i INPUT -m MODEL [-d DEVICE]
                                    [--labels LABELS] [-sr SAMPLE_RATE]
                                    [-ol OVERLAP]

Options:
  -h, --help            Show this help message and exit.
  -i INPUT, --input INPUT
                        Required. Input to process
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model.
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU or
                        GPU is acceptable. The demo
                        will look for a suitable plugin for device specified.
                        Default value is CPU
  --labels LABELS       Optional. Labels mapping file
  -sr SAMPLE_RATE, --sample_rate SAMPLE_RATE
                        Optional. Set sample rate for audio input
  -ol OVERLAP, --overlap OVERLAP
                        Optional. Set the overlapping between audio clip in
                        samples or percent
```

Running the application with the empty list of options yields the usage message given above and an error message.

You can use the following command to do inference on GPU with a pre-trained sound classification model and conversion of input audio to sample rate of 16000:

```sh
python3 sound_classification_demo.py -i <path_to_wav>/input_audio.wav -m <path_to_model>/aclnet.xml -d GPU --sample_rate 16000
```

An example audio file can be taken from https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/librispeech_s5/how_are_you_doing_today.wav.

## Demo Output

The demo uses console to display the predictions. It shows classification of each clip and total prediction of whole audio.
The demo reports

* **Latency**: total processing time required to process input data (from reading the data to displaying the results).

## See Also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/model_tools/README.md)
