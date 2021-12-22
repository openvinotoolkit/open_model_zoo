# Smartlab online Python\* Demo

This is the demo applicatio with smartlab action recognition and smartlab object detection algorithms.
The following pre-trained models are delived with the product:
`

* `smartlab-action-recognition-encoder-0001` + `smartlab-action-recognition-decoder-0001`, which are models for identifying actions of smartlab (2 actions). They recognize actions like adjust rider, put_take.

* `smartlab-object-detection-0001` + `smartlab-object-detection-0002` + `smartlab-object-detection-0003` + `smartlab-object-detection-0004`, which are models for detecing smartlab objectsi (10 objects). They detect balance, weights, tweezers, box, battery, tray, ruler, rider, scale, hand.

For more information about the pre-trained models, refer to the [Intel](../../../models/intel/index.md) and [public](../../../models/public/index.md) models documentation

## How It works

The demo pipeline consists of several steps, namely 'Decode', 'Detector', 'Segmentor', 'Evaluator', 'Display'.
Every step is implemeted by correcponding class. Decode is implemented by opencv API, Detector is imlemeted `detector.py`. Segmentor is implemented by `segmentor.py`. Evaluator is implemented by `segmentor.py`. Evaluator is implemented by `evaluator.py`. Display is implemneted by `display.py`

* `Decode` reads frames from the input videos.
* `Detector` detects smartlab objects (balance, weights, tweezers, box, battery, tray, ruler, rider scale, hand)
* `Segmentor` segment video frames based on action of the frame.
* `Evaluator` calculates scores of the current state.
* `Display` display detected objects, regnized action, calculated scores on the current frame.


> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with the `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvino.ai/latest/openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model.html#general-conversion-parameters).

## Preparing to Run
For demo input image or video files, need to provide smartlab videos following our setup (any shareable URL???).
The list of models supported by the demo is in `<omz_dir>/demos/smartlab_online_demo/python/models.lst` file.
This file can be used as a parameter for [Model Downloader](../../../tools/model_tools/README.md) and Converter to download and, if necessary, convert models to OpenVINO Inference Engine format (\*.xml + \*.bin).

An example of using the Model Downloader:

```sh
omz_downloader --list models.lst
```

An example of using the Model Converter:

```sh
omz_converter --list models.lst
```

### Supported Models
smartlab-action-recognition-encoder-0001
smartlab-action-recognition-decoder-0001
smartlab-object-detection-0001
smartlab-object-detection-0002
smartlab-object-detection-0003
smartlab-object-detection-0004

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.

## Running
TO DO

## Demo Output

The application uses OpenCV to display the real-time object detection, action recognition results and evaluation results.

## See Also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/model_tools/README.md)

# Pipeline for Online Video Analysis

The main file is `video_processor_serial.py`. In this ensemble version, given the pre-recorded videos, we aim to mimic the online process ( the results for frame k is calculated only by the historical frames)  .