# Smartlab online Python\* Demo

This is the demo application with smartlab action recognition and smartlab object detection algorithms.
This demo takes multi-view video inputs to identify actions and objects, then evaluates scores of current state.
Action recognition architecure uses two encoders for front-view and top-view respectively, and a single decoder.
Object detection uses two models for each view to detect large object and small objects, respectively.
The following pre-trained models are delived with the product:


* `smartlab-action-recognition-encoder-0001` + `smartlab-action-recognition-decoder-0001`, which are models for identifying actions of smartlab 2 actions of smartlab (adjust_rider, put_take).

* `smartlab-object-detection-0001` + `smartlab-object-detection-0002` + `smartlab-object-detection-0003` + `smartlab-object-detection-0004`, which are models for detecting smartlab objects (10 objects). They detect balance, weights, tweezers, box, battery, tray, ruler, rider, scale, hand.

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

Running the application with the `-h` option yields the following usage message:

```
usage: video_processor_serial.py [-h] -tv TOPVIEW -fv FRONTVIEW -m_ta M_TOPALL -m_tm M_TOPMOVE -m_fa M_FRONTALL -m_fm M_FRONTMOVE -m_en M_ENCODER -m_de M_DECODER

Options:
  -h, --help            Show this help message and exit.
  -tv TOPVIEW, --topview TOPVIEW
                        Required. Topview stream to be processed. The input must be a single image, a folder of images, video file or camera id.       
  -fv FRONTVIEW, --frontview FRONTVIEW
                        Required. FrontView to be processed. The input must be a single image, a folder of images, video file or camera id.
  -m_ta M_TOPALL, --m_topall M_TOPALL
                        Required. Path to topview all class model.
  -m_tm M_TOPMOVE, --m_topmove M_TOPMOVE
                        Required. Path to topview moving class model.
  -m_fa M_FRONTALL, --m_frontall M_FRONTALL
                        Required. Path to frontview all class model.
  -m_fm M_FRONTMOVE, --m_frontmove M_FRONTMOVE
                        Required. Path to frontview moving class model.
  -m_en M_ENCODER, --m_encoder M_ENCODER
                        Required. Path to encoder model.
  -m_de M_DECODER, --m_decoder M_DECODER
                        Required. Path to decoder model.
```

Running the application with an empty list of options yields the usage message given above and an error message.

**For example**, to run the demo, please provide a pathes of models, two input streams:

```sh
python3 video_processor_serial.py 
    -tv ./stream_1_top.mp4
    -fv ./stream_1_high.mp4
    -m_ta "./intel/smartlab-object-detection-0001/FP32/mw-topview-all-yolox-n.bin"
    -m_tm "./intel/smartlab-object-detection-0002/FP32/mw-topview-move-yolox-n.bin"
    -m_fa "./intel/smartlab-object-detection-0003/FP32/mw-frontview-all-yolox-n.bin"
    -m_fm "./intel/smartlab-object-detection-0004/FP32/mw-frontview-move-yolox-n.bin"
    multiview
    -m_en "./intel/smartlab-action-recognition-encoder-0001/FP32/1280vec-mobilenet-v2.bin"
    -m_de "./intel/smartlab-action-recognition-decoder-0001/FP32/concat-classifier.bin"
```

## Demo Output

The application uses OpenCV to display the real-time object detection, action recognition results and evaluation results.

## See Also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/model_tools/README.md)

# Pipeline for Online Video Analysis

The main file is `video_processor_serial.py`. In this ensemble version, given the pre-recorded videos, we aim to mimic the online process ( the results for frame k is calculated only by the historical frames)  .