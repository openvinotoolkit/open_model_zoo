# Colorization Demo

This demo demonstrates an example of using neural networks to colorize a video.
You can use the following models with the demo:

* `colorization-v2`
* `colorization-siggraph`

For more information about the pre-trained models, refer to the [model documentation](../../../models/public/index.md).

### How It Works

On the start-up, the application reads command-line parameters and loads one network to the Inference Engine for execution.

Once the program receives an image, it performs the following steps:

1. Converts the frame of video into the LAB color space.
2. Uses the L-channel to predict A and B channels.
3. Restores the image by converting it into the BGR color space.

### Running the Demo

Running the application with the `-h` option yields the following usage message:

```
usage: colorization_demo.py [-h] -m MODEL [-d DEVICE] -i INPUT [--loop]
                            [-o OUTPUT] [-limit OUTPUT_LIMIT]
                            [--no_show] [-v] [-u UTILIZATION_MONITORS]

Options:
  -h, --help            Help with the script.
  -m MODEL, --model MODEL
                        Required. Path to .xml file with pre-trained model.
  -d DEVICE, --device DEVICE
                        Optional. Specify target device for infer: CPU, GPU,
                        FPGA, HDDL or MYRIAD. Default: CPU
  -i INPUT, --input INPUT
                        Required. An input to process. The input must be a single image,
                        a folder of images, video file or camera id.
  --loop                Optional. Enable reading the input in a loop.
  -o OUTPUT, --output OUTPUT
                        Optional. Name of output to save.
  -limit OUTPUT_LIMIT, --output_limit OUTPUT_LIMIT
                        Optional. Number of frames to store in output.
                        If 0 is set, all frames are stored.
  --no_show             Optional. Don't show output.
  -v, --verbose         Optional. Enable display of processing logs on screen.
  -u UTILIZATION_MONITORS, --utilization_monitors UTILIZATION_MONITORS
                        Optional. List of monitors to show initially.


```

To run the demo, you can use public or Intel's pretrained models. To download pretrained models, use the OpenVINO&trade; [Model Downloader](../../../tools/downloader/README.md). The list of models supported by the demo is in `<omz_dir>/demos/colorization_demo/python/models.lst`.

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

### Demo Output

The demo uses OpenCV to display the colorized frame.

## See Also

* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
