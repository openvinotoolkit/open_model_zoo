# VA Object Detection C++ Demo

![](../object_detection.gif)

This demo showcases HW-accelerated video decoding using libVA and sharing VA surface data with input blob without copying.
Data is then processed inside Inference Engine using SSD-like object detection models.
Image data sharing can improve overall frame-rate of the application, because it removes data copying process.
Note that due to demo specifics it work only on GPU device and only on Linux platform.

Other demo objectives are:

* Video as input support via OpenCV
* Visualization of the resulting bounding boxes and class number
* OpenCV is used to draw resulting bounding boxes, labels, so you can copy paste this code without
need to pull Inference Engine demos helpers to your app
* Demonstration of the VA shared context capabilities

## Running

Running the application with the `-h` option yields the following usage message:

```
InferenceEngine:
    API version ............ <version>
    Build .................. <number>

va\_object\_detection\_demo [OPTION]
Options:

    -h                        Print a usage message.
    -i                        Required. An input to process. The input must be a single image, a folder of images, video file or camera id.
    -m "<path>"               Required. Path to an .xml file with a trained model.
    -o "<path>"               Optional. Name of output to save.
    -limit "<num>"            Optional. Number of frames to store in output. If 0 is set, all frames are stored.
      -c "<absolute_path>"    Required for GPU custom kernels. Absolute path to the .xml file with the kernel descriptions.
    -pc                       Optional. Enables per-layer performance report.
    -r                        Optional. Inference results as raw values.
    -t                        Optional. Probability threshold for detections.
    -loop                     Optional. Enable reading the input in a loop.
    -no_show                  Optional. Don't show output.
    -output_resolution        Optional. Specify the maximum output window resolution in (width x height) format. Example: 1280x720. Input frame size used by default.
    -u                        Optional. List of monitors to show initially.
```

Running the application with the empty list of options yields the usage message given above and an error message.

To run the demo, you can use public or pre-trained models. To download the pre-trained models, use the OpenVINO [Model Downloader](../../../tools/downloader/README.md). The list of models supported by the demo is in `<omz\_dir>/demos/va\_object\_detection_demo/cpp/models.lst`.

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

If labels file is used, it should correspond to model output. Demo suggests labels listed in the file to be indexed from 0, one line - one label (i.e. very first line contains label for ID 0). Note that some models may return labels IDs in range 1..N, in this case label file should contain "background" label at the very first line.

You can use the following command to do inference with a pre-trained object detection model:
```sh
./va_object_detection_demo -i <path_to_video>/inputVideo.mp4 -at ssd -m <path_to_model>/ssd.xml
```

## Demo Output

The demo uses OpenCV to display the resulting frame with detections (rendered as bounding boxes and labels, if provided).
The demo reports:

* **FPS**: average rate of video frame processing (frames per second).
* **Latency**: average time required to process one frame (from reading the frame to displaying the results).
* **Avg Time**: average processing time for each stage.
You can use these metrics to measure application-level performance.


## See Also
* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
