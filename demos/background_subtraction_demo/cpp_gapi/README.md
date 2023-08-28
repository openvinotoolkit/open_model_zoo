# G-API Background Subtraction Demo

This demo shows how to perform background subtraction using G-API.

> **NOTE**: Only batch size of 1 is supported.

## How It Works
The demo application expects an instance-segmentation-security-???? or trimap free background matting based on pixel-level segmentation approach model in the Intermediate Representation (IR) format. Please note, that there aren't background matting models in `OpenModelZoo` collection.

1. for instance segmentation models based on `Mask RCNN` approach:
    * One input: `image` for input image.
    * At least three outputs including:
        * `boxes` with absolute bounding box coordinates of the input image and its score
        * `labels` with object class IDs for all bounding boxes
        * `masks` with fixed-size segmentation heat maps for all classes of all bounding boxes
2. for tripmap free background matting based on pixel-level segmentation approach:
    * Single 1x3xWxH input.
    * Single 1x1xWxH output - float tensor which is alpha channel for input.

The use case for the demo is an online conference where is needed to show only foreground - people and, respectively, to hide or replace background.

As input, the demo application accepts a path to a single image file, a video file or a numeric ID of a web camera specified with a command-line argument `-i`

The demo workflow is the following:

1. The demo application reads image/video frames one by one, resizes them to fit into the input image blob of the network (`image`).
2. The demo visualizes the resulting background subtraction. Certain command-line options affect the visualization:
    * If you specify `--target_bgr`, background will be replaced by a chosen image or video. By default background replaced by green field.
    * If you specify `--blur_bgr`, background will be blurred according to a set value. By default equal to zero and is not applied.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with the `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Embedding Preprocessing Computation](@ref openvino_docs_MO_DG_Additional_Optimization_Use_Cases).

## Preparing to Run

For demo input image or video files, refer to the section **Media Files Available for Demos** in the [Open Model Zoo Demos Overview](../../README.md).
The list of models supported by the demo is in `<omz_dir>/demos/background_subtraction_demo/cpp_gapi/models.lst` file.
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

* instance-segmentation-person-????

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.


### OneVPL Support

Demo provides functionality to use [OneVPL](https://github.com/oneapi-src/oneVPL#-video-processing-library) video decoding.
Example:
```sh
./background_subtraction_demo_gapi/ -m <path_to_model> -i <path_to_video_file> -use_onevpl
```

In order to provide additional configuration paramaters use `-onevpl_params`:
```sh
./background_subtraction_demo_gapi/ -m <path_to_model> -i <path_to_raw_file> -use_onevpl -onevpl_params="mfxImplDescription.mfxDecoderDescription.decoder.CodecID:MFX_CODEC_HEVC"
```
>**NOTE**: Only raw formats such as `h264`, `h265` etc are supported on Linux.
Working with raw formats user always must specify `codec` type via `-onevpl_params`. See example below.

To build OpenCV G-API with `oneVPL` support follow instruction:
[Building G-API with oneVPL Toolkit support](https://github.com/opencv/opencv/wiki/Graph-API#building-with-onevpl-toolkit-support)

#### Troubleshooting
During execution `oneVPL` might report warnings that tell the user that source can be configurable more accurate.

For example:
```
 cv::gapi::wip::onevpl::VPLLegacyDecodeEngine::process_error [000001CED3851C70] error: cv::gapi::wip::onevpl::CachedPool::find_free - cannot get free surface from pool, size: 5
```
This might be fixed by increasing pool size using `-onevpl_pool_size` parameter.

## Running

Run the application with the `-h` option to see the following usage message:

```
[ INFO ] OpenVINO Runtime version ......... <version>
[ INFO ] Build ........... <build>

background_subtraction_demo_gapi [OPTION]
Options:

    -h                         Print a usage message.
    -i                         Required. An input to process. The input must be a single image, a folder of images, video file or camera id.
    -loop                      Optional. Enable reading the input in a loop.
    -o "<path>"                Optional. Name of the output file(s) to save. Frames of odd width or height can be truncated. See https://github.com/opencv/opencv/pull/24086
    -limit "<num>"             Optional. Number of frames to store in output. If 0 is set, all frames are stored.
    -res "<WxH>"               Optional. Set camera resolution in format WxH.
    -at "<type>"               Required. Architecture type: maskrcnn.
    -m "<path>"                Required. Path to an .xml file with a trained model.
    -kernel_package "<string>" Optional. G-API kernel package type: opencv, fluid (by default opencv is used).
    -d "<device>"              Optional. Target device for network (the list of available devices is shown below). The demo will look for a suitable plugin for a specified device. Default value is "CPU".
    -nireq "<integer>"         Optional. Number of infer requests. If this option is omitted, number of infer requests is determined automatically.
    -nthreads "<integer>"      Optional. Number of threads.
    -nstreams                  Optional. Number of streams to use for inference on the CPU or/and GPU in throughput mode (for HETERO and MULTI device cases use format <device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>)
    -no_show                   Optional. Don't show output.
    -blur_bgr                  Optional. Blur background.
    -target_bgr                Optional. Background onto which to composite the output (by default to green field).
    -u                         Optional. List of monitors to show initially.
    -use_onevpl                Optional. Use onevpl video decoding.
    -onevpl_params             Optional. Parameters for onevpl video decoding. OneVPL source can be fine-grained by providing configuration parameters. Format: <prop name>:<value>,<prop name>:<value> Several important configuration parameters: 'mfxImplDescription.mfxDecoderDescription.decoder.CodecID' values: https://spec.oneapi.io/onevpl/2.7.0/API_ref/VPL_enums.html?highlight=mfx_codec_hevc#codecformatfourcc and 'mfxImplDescription.AccelerationMode' values: https://spec.oneapi.io/onevpl/2.7.0/API_ref/VPL_disp_api_enum.html?highlight=d3d11#mfxaccelerationmode(see `MFXSetConfigFilterProperty` by https://spec.oneapi.io/versions/latest/elements/oneVPL/source/index.html)
    -onevpl_pool_size          OneVPL source applies this parameter as preallocated frames pool size. 0 leaves frames pool size default for your system. This parameter doesn't have a god default value. It must be adjusted for specific execution (video, model, system ...).

Available target devices:  <targets>
```

Running the application with an empty list of options yields the short version of the usage message and an error message.

To run the demo, please provide paths to the model in the IR format, and to an input video, image, or folder with images:

```bash
./background_subtraction_demo_gapi/ -m <path_to_model> -i <path_to_file>
```

>**NOTE**: If you provide a single image as an input, the demo processes and renders it quickly, then exits. To continuously visualize inference results on the screen, apply the `loop` option, which enforces processing a single image in a loop.

You can save processed results to a Motion JPEG AVI file or separate JPEG or PNG files using the `-o` option:

* To save processed results in an AVI file, specify the name of the output file with `avi` extension, for example: `-o output.avi`.
* To save processed results as images, specify the template name of the output image file with `jpg` or `png` extension, for example: `-o output_%03d.jpg`. The actual file names are constructed from the template at runtime by replacing regular expression `%03d` with the frame number, resulting in the following: `output_000.jpg`, `output_001.jpg`, and so on.
To avoid disk space overrun in case of continuous input stream, like camera, you can limit the amount of data stored in the output file(s) with the `limit` option. The default value is 1000. To change it, you can apply the `-limit N` option, where `N` is the number of frames to store.

>**NOTE**: Windows\* systems may not have the Motion JPEG codec installed by default. If this is the case, you can download OpenCV FFMPEG back end using the PowerShell script provided with the OpenVINO &trade; install package and located at `<INSTALL_DIR>/opencv/ffmpeg-download.ps1`. The script should be run with administrative privileges if OpenVINO &trade; is installed in a system protected folder (this is a typical case). Alternatively, you can save results as images.

## Demo Output

The application uses OpenCV to display resulting images.
The demo reports

* **FPS**: average rate of video frame processing (frames per second).

## See Also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/model_tools/README.md)
