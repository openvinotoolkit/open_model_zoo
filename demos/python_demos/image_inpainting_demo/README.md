# Image Inpainting Python Demo

This demo showcases Image Inpainting with GMCNN. The task is to estimate suitable pixel information
to fill holes in images.

## How It Works
This demo can work in 2 modes: 

* GUI mode: areas for inpainting can be marked interactively using mouse painting
* Auto mode (use -ac or -ar option for it): image will be processed automatically using randomly applied mask (-ar option) or using specific color-based mask (-ac option)

Running the application with the `-h` option yields the following usage message:

```
usage: image_inpainting_demo.py [-h] -m MODEL [-i INPUT] [-d DEVICE]
                                [-p PARTS] [-mbw MAX_BRUSH_WIDTH]
                                [-ml MAX_LENGTH] [-mv MAX_VERTEX] [--no_show]
                                [-o OUTPUT] [-ac C C C] [-ar]

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
                        Optional. Number of parts to draw mask. Ignored in GUI
                        mode
  -mbw MAX_BRUSH_WIDTH, --max_brush_width MAX_BRUSH_WIDTH
                        Optional. Max width of brush to draw mask. Ignored in
                        GUI mode
  -ml MAX_LENGTH, --max_length MAX_LENGTH
                        Optional. Max strokes length to draw mask. Ignored in
                        GUI mode
  -mv MAX_VERTEX, --max_vertex MAX_VERTEX
                        Optional. Max number of vertex to draw mask. Ignored
                        in GUI mode
  --no_show             Optional. Don't show output. Cannot be used in GUI
                        mode
  -o OUTPUT, --output OUTPUT
                        Optional. Save output to the file with provided
                        filename. Ignored in GUI mode
  -ac C C C, --auto_mask_color C C C
                        Optional. Use automatic (non-interactive) mode with
                        color mask.Provide color to be treated as mask (3 RGB
                        components in range of 0...255). Cannot be used
                        together with -ar.
  -ar, --auto_mask_random
                        Optional. Use automatic (non-interactive) mode with
                        random mask for inpainting (with parameters set by -p,
                        -mbw, -mk and -mv). Cannot be used together with -ac.
```

To run the demo, you can use public or pretrained models. You can download the pretrained models with the OpenVINO&trade; [Model Downloader](../../../tools/downloader/README.md).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

## GUI Mode operation
In GUI mode user can draw mask using mouse (holding left mouse button). The brush size is adjustable using slider on the top of the screen. After the mask painting is done, inpainting processing can be started by pressing Space or Enter key.

Also, these hot keys are available:

* **Backspace or C** to clear current mask
* **Space or Enter** to inpaint
* **R** to reset all changes
* **Tab** to show original image
* **Esc or Q** to quit

## Demo Output

In auto mode this demo uses OpenCV to display the resulting image and image with mask applied and reports performance in the format of summary inference FPS. Processed image can be also written to file.

In GUI mode this demo provides interactive means to apply mask and see the result of processing instantly (see hotkeys above).

## See Also

* [Using Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
