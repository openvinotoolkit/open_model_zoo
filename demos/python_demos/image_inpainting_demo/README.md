# Image Inpainting Python Demo

This demo showcases Image Inpainting with GMCNN. The task is to estimate suitable pixel information
to fill holes in images.

## How It Works

Running the application with the `-h` option yields the following usage message:

```
usage: image_inpainting_demo.py [-h] -m MODEL [-i INPUT] [-d DEVICE]
                                [-p PARTS] [-mbw MAX_BRUSH_WIDTH]
                                [-ml MAX_LENGTH] [-mv MAX_VERTEX] [--no_show]

Options:
  -h, --help            Show this help message and exit.
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model.
  -i INPUT, --input INPUT
                        path to image.
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU,
                        GPU, FPGA, HDDL or MYRIAD is acceptable. The demo will
                        look for a suitable plugin for device specified.
                        Default value is CPU
  -p PARTS, --parts PARTS
                        Optional. Number of parts to draw mask.
  -mbw MAX_BRUSH_WIDTH, --max_brush_width MAX_BRUSH_WIDTH
                        Optional. Max width of brush to draw mask.
  -ml MAX_LENGTH, --max_length MAX_LENGTH
                        Optional. Max strokes length to draw mask.
  -mv MAX_VERTEX, --max_vertex MAX_VERTEX
                        Optional. Max number of vertex to draw mask.
  --no_show             Optional. Don't show output
```

To run the demo, you can use public or pretrained models. You can download the pretrained models with the OpenVINO&trade; [Model Downloader](../../../tools/downloader/README.md).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

## Demo Output

The demo uses OpenCV to display the resulting image and image with mask applied and reports performance in the format of summary inference FPS.

## See Also

* [Using Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
