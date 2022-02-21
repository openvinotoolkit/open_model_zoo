# Smartlab Python\* Demo

This is the demo application with smartlab action recognition and smartlab object detection algorithms.
This demo takes multi-view video inputs to identify actions and objects, then evaluates scores of current state.
Action recognition architecure uses two encoders for front-view and top-view respectively, and a single decoder.
Object detection uses two models for each view to detect large object and small objects, respectively.
The following pre-trained models are delivered with the product:

* `i3d-rgb-tf` + `smartlab-sequence-modelling-0001`, which are other models for identifying actions 2 actions of smartlab (adjust_rider, put_take).

* `smartlab-object-detection-0001` + `smartlab-object-detection-0002` + `smartlab-object-detection-0003` + `smartlab-object-detection-0004`, which are models for detecting smartlab objects (10 objects). They detect balance, weights, tweezers, box, battery, tray, ruler, rider, scale, hand.

## How It works

The demo pipeline consists of several steps:

* `Decode` reads frames from the input videos
* `Detector` detects smartlab objects (balance, weights, tweezers, box, battery, tray, ruler, rider scale, hand)
* `Segmentor` segments video frames based on action of the frame
* `Evaluator` calculates scores of the current state
* `Display` displays detected objects, recognized action, calculated scores on the current frame


> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with the `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvino.ai/latest/openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model.html#general-conversion-parameters).

## Preparing to Run
For demo input image or video files, you need to provide smartlab videos (https://storage.openvinotoolkit.org/data/test_data/videos/smartlab/).
The list of models supported by the demo is in `<omz_dir>/demos/smartlab_demo/python/models.lst` file.
This file can be used as a parameter for [Model Downloader](../../../tools/model_tools/README.md) and Converter to download and, if necessary, convert models to OpenVINO format (\*.xml + \*.bin).

An example of using the Model Downloader:

```sh
omz_downloader --list models.lst
```

### Supported Models
* smartlab-object-detection-0001
* smartlab-object-detection-0002
* smartlab-object-detection-0003
* smartlab-object-detection-0004
* smartlab-sequence-modelling-0001
* i3d-rgb-tf

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.

## Running

Running the application with an empty list of options yields the usage message given above and an error message.

**For example**, to run the demo, please provide the model paths and two input streams:

```sh
python3 smartlab_demo.py
    -tv ./stream_1_top.mp4
    -fv ./stream_1_high.mp4
    -m_ta "./intel/smartlab-object-detection-0001/FP32/smartlab-object-detection-0001.xml"
    -m_tm "./intel/smartlab-object-detection-0002/FP32/smartlab-object-detection-0002.xml"
    -m_fa "./intel/smartlab-object-detection-0003/FP32/smartlab-object-detection-0003.xml"
    -m_fm "./intel/smartlab-object-detection-0004/FP32/smartlab-object-detection-0004.xml"
    -m_en "./public/i3d-rgb-tf/FP32/i3d-rgb-tf.xml"
    -m_de "./intel/smartlab-sequence-modelling-0001/FP32/smartlab-sequence-modelling-0001.xml"
```

## Demo Output

The application uses OpenCV to display the real-time object detection, action recognition results and evaluation results.

## See Also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/model_tools/README.md)
