# Sound Classification Python\* Demo

Demo application for sound classification algorithm.

## How It Works

Upon the start-up the demo application reads command line parameters and loads a network to Inference engine. It uses only audiofiles in `wav` format. Audio should be converted to model's sample rate using `-sr/--samplerate` option, if sample rate of audio differs from sample rate of model (e.g. [AclNet](../../../model/public/aclnet/aclnet.md) expected 16kHz audio). After reading the audio, it is sliced into clips to fit model input (clips are allowed to overlap with `-ol/--overlap` option) and each clip is processed separately with its own prediction. 

## Running

Run the application with the `-h` option to see the usage message:
```
python3 audio_classification_demo.py -h
```
The command yields the following usage message:
```
usage: audio_classification_demo.py [-h] -i INPUT -m MODEL [-l CPU_EXTENSION]
                                    [-d DEVICE] [--labels LABELS]
                                    [-sr SAMPLERATE] [-ol OVERLAP]

Options:
  -h, --help            Show this help message and exit.
  -i INPUT, --input INPUT
                        Required. Input to process
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model.
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        Optional. Required for CPU custom layers. Absolute
                        path to a shared library with the kernels
                        implementations.
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU,
                        GPU, FPGA, HDDL or MYRIAD is acceptable. The sample
                        will look for a suitable plugin for device specified.
                        Default value is CPU
  --labels LABELS       Optional. Labels mapping file
  -sr SAMPLERATE, --sample_rate SAMPLERATE
                        Optional. Set sample rate for audio input
  -ol OVERLAP, --overlap OVERLAP
                        Optional. Set the overlapping between audio clip in
                        samples or percent
```
Running the application with the empty list of options yields the usage message given above and an error message.
You can use the following command to do inference on GPU with a pre-trained sound classification model and conversion of input audio to samplerate of 16000:
```
python3 audio_classification_demo.py -i <path_to_wav>/input_audio.wav -m <path_to_model>/aclnet.xml -d GPU --samplerate 16000
```

To run the demo, you can use public or pre-trained models. You can download the pre-trained models with the OpenVINO [Model Downloader](../../../tools/downloader/README.md) or from [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

## Demo Output

The demo uses console to display the predictions. It shows classification of each clip with timing of it and total prediction of whole audio.

## See Also
* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
