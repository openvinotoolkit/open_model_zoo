# Gesture Recognition Python\* Demo

![example](./gesture_recognition.gif)

This demo demonstrates how to run Gesture (e.g. American Sign Language (ASL) gestures) Recognition models using OpenVINO&trade; toolkit.

## How It Works

The demo application expects an gesture recognition model in the Intermediate Representation (IR) format.

As input, the demo application takes:

* a path to a video file or a device node of a web-camera specified with a command line argument `--input`
* a path to a file in JSON format with gesture class names `--class_map`

The demo workflow is the following:

1. The demo application reads video frames one by one, runs person detector that extracts ROI, tracks the ROI of very first person. Additional process is used to prepare the batch of frames with constant framerate.
2. Batch of frames and extracted ROI are passed to artificial neural network that predicts the gesture.
3. The app visualizes results of its work as graphical window where following objects are shown:
    - Input frame with detected ROI.
    - Last recognized gesture.
    - Performance characteristics.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

## Preparing to run

For demo input image or video files you may refer to [Media Files Available for Demos](../../README.md#Media-Files-Available-for-Demos).
Pre-trained models, supported by demo listed in [models.lst](./models.lst) file, located at each demo folder.
This file can be used as a parameter for [Model Downloader](../../../tools/downloader/README.md) and Converter to download and, if necessary, convert models to OpenVINO Inference Engine format (\*.xml + \*.bin).

### Supported models

* asl-recognition-0004
* common-sign-language-0001
* person-detection-asl-0001

> **NOTE**: Refer to tables for [Intel](../../../models/intel/device_support.md) and [public](../../../models/public/device_support.md) models which summarize models support at different devices to select target inference device.

## Running

Run the application with the `-h` option to see the following usage message:

```
usage: gesture_recognition_demo.py [-h] -m_a ACTION_MODEL -m_d DETECTION_MODEL
                                   -i INPUT [-o OUTPUT] [-limit OUTPUT_LIMIT]
                                   -c CLASS_MAP [-s SAMPLES_DIR]
                                   [-t ACTION_THRESHOLD] [-d DEVICE]
                                   [-l CPU_EXTENSION] [--no_show]
                                   [-u UTILIZATION_MONITORS]

Options:
  -h, --help            Show this help message and exit.
  -m_a ACTION_MODEL, --action_model ACTION_MODEL
                        Required. Path to an .xml file with a trained gesture
                        recognition model.
  -m_d DETECTION_MODEL, --detection_model DETECTION_MODEL
                        Required. Path to an .xml file with a trained person
                        detector model.
  -i INPUT, --input INPUT
                        Required. Path to a video file or a device node of a
                        web-camera.
  -o OUTPUT, --output OUTPUT
                        Optional. Name of output to save.
  -limit OUTPUT_LIMIT, --output_limit OUTPUT_LIMIT
                        Optional. Number of frames to store in output.
                        If 0 is set, all frames are stored.
  -c CLASS_MAP, --class_map CLASS_MAP
                        Required. Path to a file with gesture classes.
  -s SAMPLES_DIR, --samples_dir SAMPLES_DIR
                        Optional. Path to a directory with video samples of
                        gestures.
  -t ACTION_THRESHOLD, --action_threshold ACTION_THRESHOLD
                        Optional. Threshold for the predicted score of an
                        action.
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on: CPU,
                        GPU, FPGA, HDDL or MYRIAD. The demo will look for a
                        suitable plugin for device specified (by default, it
                        is CPU).
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        Optional. Required for CPU custom layers. Absolute
                        path to a shared library with the kernels
                        implementations.
  --no_show             Optional. Do not visualize inference results.
  -u UTILIZATION_MONITORS, --utilization_monitors UTILIZATION_MONITORS
                        Optional. List of monitors to show initially.
```

Running the application with an empty list of options yields the short version of the usage message and an error message.

To run the demo, please provide paths to the gesture recognition and person detection models in the IR format, to a file with class names, and to an input video:

```bash
python gesture_recognition_demo.py \
    -m_a <path_to_model>/asl-recognition-0004.xml \
    -m_d <path_to_model>/person-detection-asl-0001.xml \
    -i 0 \
    -c <omz_dir>/data/dataset_classes/msasl100.json
```

The demo starts in person tracking mode and to switch it in the action recognition mode you should press `0-9` button with appropriate detection ID (the number in top-left of each bounding box). If frame contains only one person, they will be chosen automatically. After that you can switch back to tracking mode by pressing space button.

An example of file with class names can be found within the OMZ directory at `<omz_dir>/data/dataset_classes/msasl100.json`.

> **NOTE**: To run the demo application with video examples of gestures specify the `-s` key with valid path to the directory with video samples. The name of each video sample should be the valid name of gesture from `<omz_dir>/data/dataset_classes/msasl100.json` file. To navigate between samples use 'f' and 'b' keys for iterating next and previous respectively video sample.

## Demo Output

The application uses OpenCV to display gesture recognition result and current inference performance.

## See Also

* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
