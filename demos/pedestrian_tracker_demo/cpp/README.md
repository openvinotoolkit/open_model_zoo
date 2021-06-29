# Pedestrian Tracker C++ Demo

![example](./pedestrian_tracker.gif)

This demo showcases Pedestrian Tracking scenario: it reads frames from an input video sequence, detects pedestrians in the frames, and builds trajectories of movement of the pedestrians in a frame-by-frame manner.

## How It Works

On startup, the application reads command line parameters and loads the specified networks.

Upon getting a frame from the input video sequence (either a video file or a folder with images), the app performs inference of the pedestrian detector network.

After that, the bounding boxes describing the detected pedestrians are passed to the instance of the tracker class that matches the appearance of the pedestrians with the known
(already tracked) persons.
In obvious cases (when pixel-to-pixel similarity of a detected pedestrian is sufficiently close to the latest pedestrian image from one of the known tracks),
the match is made without inference of the reidentification network. In more complicated cases, the demo uses the reidentification network to make a decision
if a detected pedestrian is the next position of a known person or the first position of a new tracked person.

After that, the application displays the tracks and the latest detections on the screen and goes to the next frame.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with the `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

## Preparing to Run

For demo input image or video files you may refer to [Media Files Available for Demos](../../README.md#Media-Files-Available-for-Demos).
The list of models supported by the demo is in `<omz_dir>/demos/pedestrian_tracker_demo/cpp/models.lst` file.
This file can be used as a parameter for [Model Downloader](../../../tools/downloader/README.md) and Converter to download and, if necessary, convert models to OpenVINO Inference Engine format (\*.xml + \*.bin).

An example of using the Model Downloader:

```sh
python3 <omz_dir>/tools/downloader/downloader.py --list models.lst
```

An example of using the Model Converter:

```sh
python3 <omz_dir>/tools/downloader/converter.py --list models.lst
```

### Supported Models

* person-detection-retail-0002
* person-detection-retail-0013
* person-reidentification-retail-0277
* person-reidentification-retail-0286
* person-reidentification-retail-0287
* person-reidentification-retail-0288

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.

## Running

Running the application with the `-h` option yields the following usage message:

```
InferenceEngine:
    API version ............ <version>
    Build .................. <number>

pedestrian_tracker_demo [OPTION]
Options:

    -h                           Print a usage message.
    -i                           Required. An input to process. The input must be a single image, a folder of images, video file or camera id.
    -loop                        Optional. Enable reading the input in a loop.
    -first                       Optional. The index of the first frame of the input to process. The actual first frame captured depends on cv::VideoCapture implementation and may have slightly different number.
    -read_limit                  Optional. Read length limit before stopping or restarting reading the input.
    -o "<path>"                  Optional. Name of the output file(s) to save.
    -limit "<num>"               Optional. Number of frames to store in output. If 0 is set, all frames are stored.
    -m_det "<path>"              Required. Path to the Pedestrian Detection Retail model (.xml) file.
    -m_reid "<path>"             Required. Path to the Pedestrian Reidentification Retail model (.xml) file.
    -l "<absolute_path>"         Optional. For CPU custom layers, if any. Absolute path to a shared library with the kernels implementation.
          Or
    -c "<absolute_path>"         Optional. For GPU custom kernels, if any. Absolute path to the .xml file with the kernels description.
    -d_det "<device>"            Optional. Specify the target device for pedestrian detection (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin.
    -d_reid "<device>"           Optional. Specify the target device for pedestrian reidentification (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin.
    -r                           Optional. Output pedestrian tracking results in a raw format (compatible with MOTChallenge format).
    -pc                          Optional. Enable per-layer performance statistics.
    -no_show                     Optional. Don't show output.
    -delay                       Optional. Delay between frames used for visualization. If negative, the visualization is turned off (like with the option 'no_show'). If zero, the visualization is made frame-by-frame.
    -out "<path>"                Optional. The file name to write output log file with results of pedestrian tracking. The format of the log file is compatible with MOTChallenge format.
    -u                           Optional. List of monitors to show initially.
```

For example, to run the application with the OpenVINO&trade; toolkit pre-trained models with inferencing pedestrian detector on a GPU and pedestrian reidentification on a CPU, run the following command:

```sh
./pedestrian_tracker_demo -i <path_video_file> \
                          -m_det <path_to_model>/person-detection-retail-0013.xml \
                          -m_reid <path_to_model>/person-reidentification-retail-0277.xml \
                          -d_det GPU
```

>**NOTE**: If you provide a single image as an input, the demo processes and renders it quickly, then exits. To continuously visualize inference results on the screen, apply the `loop` option, which enforces processing a single image in a loop.

You can save processed results to a Motion JPEG AVI file or separate JPEG or PNG files using the `-o` option:

* To save processed results in an AVI file, specify the name of the output file with `avi` extension, for example: `-o output.avi`.
* To save processed results as images, specify the template name of the output image file with `jpg` or `png` extension, for example: `-o output_%03d.jpg`. The actual file names are constructed from the template at runtime by replacing regular expression `%03d` with the frame number, resulting in the following: `output_000.jpg`, `output_001.jpg`, and so on.
To avoid disk space overrun in case of continuous input stream, like camera, you can limit the amount of data stored in the output file(s) with the `limit` option. The default value is 1000. To change it, you can apply the `-limit N` option, where `N` is the number of frames to store.

>**NOTE**: Windows\* systems may not have the Motion JPEG codec installed by default. If this is the case, you can download OpenCV FFMPEG back end using the PowerShell script provided with the OpenVINO &trade; install package and located at `<INSTALL_DIR>/opencv/ffmpeg-download.ps1`. The script should be run with administrative privileges if OpenVINO &trade; is installed in a system protected folder (this is a typical case). Alternatively, you can save results as images.

## Demo Output

The demo uses OpenCV to display the resulting frame with detections rendered as bounding boxes, curves (for trajectories displaying), and text.

## See Also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
