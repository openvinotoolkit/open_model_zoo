# Speech Recognition Offline Demo

This demo demonstrates Automatic Speech Recognition (ASR) with pretrained QuartzNet model.

## How It Works

After computing audio features, running a neural network to get character probabilities, and CTC greedy decoding, the demo prints the decoded text.

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
```
python speech_recognition_offline_demo.py -m quartznet-15x5-en.xml -i audio.wav
```

**Only 16-bit, 16 kHz, mono-channel WAVE audio files are supported.**

An example audio file can be taken from `<openvino_dir>/deployment_tools/demo/how_are_you_doing.wav`.

## Demo Output

The application prints the decoded text for the audio file.
