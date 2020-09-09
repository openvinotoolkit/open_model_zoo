# Image Inpainting Python Demo

This demo showcases Image Inpainting with GMCNN. The task is to estimate suitable pixel information
to fill holes in images.

## How It Works
This demo can work in 2 modes: 
* GUI mode: areas for inpainting can be marked interactively using mouse painting
* Auto mode (use -a option for it): image will be processed automatically using randomly applied mask (-r option) or using specific color-based mask (-mc option to set mask color)

Running the application with the `-h` option yields the following usage message:

```
usage: image_inpainting_demo.py [-h] -m MODEL [-i INPUT] [-d DEVICE] [-r]
                                [-p PARTS] [-mbw MAX_BRUSH_WIDTH]
                                [-ml MAX_LENGTH] [-mv MAX_VERTEX]
                                [-mc MASK_COLOR [MASK_COLOR ...]] [--no_show]
                                [-o OUTPUT] [-a]
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
  -r, --rnd             Optional. Use random mask for inpainting (with
                        parameters set by -p, -mbw, -mk and -mv).Skipped in
                        GUI mode
  -p PARTS, --parts PARTS
                        Optional. Number of parts to draw mask. Skipped in GUI
                        mode
  -mbw MAX_BRUSH_WIDTH, --max_brush_width MAX_BRUSH_WIDTH
                        Optional. Max width of brush to draw mask. Skipped in
                        GUI mode
  -ml MAX_LENGTH, --max_length MAX_LENGTH
                        Optional. Max strokes length to draw mask. Skipped in
                        GUI mode
  -mv MAX_VERTEX, --max_vertex MAX_VERTEX
                        Optional. Max number of vertex to draw mask. Skipped
                        in GUI mode
  -mc MASK_COLOR [MASK_COLOR ...], --mask_color MASK_COLOR [MASK_COLOR ...]
                        Optional. Color to be treated as mask (provide 3 RGB
                        components in range of 0...255). Default is 0 0 0.
                        Skipped in GUI mode
  --no_show             Optional. Don't show output. Cannot be used in GUI mode
  -o OUTPUT, --output OUTPUT
                        Optional. Save output to the file with provided
                        filename. Skipped in GUI mode
  -a, --auto            Optional. Use automatic (non-interactive) mode instead
                        of GUI
```

To run the demo, you can use public or pretrained models. You can download the pretrained models with the OpenVINO&trade; [Model Downloader](../../../tools/downloader/README.md).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

## Demo Output

The demo uses OpenCV to display the resulting image and image with mask applied and reports performance in the format of summary inference FPS.

## See Also

* [Using Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
