# Colorization Demo

This demo demonstrates an example of using neural networks to colorize a video.
You can use the following models with the demo:

* `colorization-v2`
* `colorization-v2-norebal`

For more information about the pre-trained models, refer to the [model documentation](../../../models/public/index.md).

### How it works

On the start-up, the application reads command line parameters and loads one network to the Inference Engine for execution.

Having received the image, the program:
1) converts the frame of video into LAB color space
2) uses the L-channel to predict A and B channels
3) restores the image, by converting it into BGR color space

### Running the demo:

Running the application with the `-h` option yields the following usage message:

```
colorization_demo.py -h

usage: colorization_demo.py [-h] -m MODEL --coeffs COEFFS [-d DEVICE] -i
                            "<path>" [--no_show] [-v]

Options:
  -h, --help            Help with the script.
  -m MODEL, --model MODEL
                        Required. Path to .xml file with pre-trained model.
  --coeffs COEFFS       Required. Path to .npy file with color coefficients.
  -d DEVICE, --device DEVICE
                        Optional. Specify target device for infer: CPU, GPU,
                        FPGA, HDDL or MYRIAD. Default: CPU
  -i "<path>", --input "<path>"
                        Required. Input to process.
  --no_show             Optional. Disable display of results on screen.
  -v, --verbose         Optional. Enable display of processing logs on screen.
```

To run the demo, you can use public or pre-trained models. To download the pre-trained models, use the OpenVINO [Model Downloader](../../../tools/downloader/README.md) or go to [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

### Demo output

The demo uses OpenCV to display the colorized frame.

## See also

* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
