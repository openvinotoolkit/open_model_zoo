# Denoise Ray Tracing Demo

This example demonstrates an approach to denoising which is suitable for images rendered with Monte Carlo ray tracing methods like unidirectional and bidirectional path tracing using OpenVINO™.

This demo also supports images from datasets [Tungsten](https://sites.google.com/view/bilateral-grid-denoising/home/supplemental-material-dataset).

## How It Works

The demo workflow is the following:

The demo first reads an image and performs the preprocessing such as autoexposure and padding. Then after loading model to the plugin, the inference will start. The demo will display the image.

## Preparing to Run

The list of models supported by the demo is in `<omz_dir>/demos/denoise_ray_tracing_demo/python/models.lst` file.
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

* denoise_rt_hdr_alb

## Running

Running the application with the `-h` option yields the following usage message:

```
usage: denoise_ray_traicing_demo.py [-h] -m MODEL --hdr HDR --albedo ALBEDO
                                     [-d DEVICE] [--no_show] [-o OUTPUT]
                                     [-ob OUTPUT_BLOB]
                                     [--input_scale INPUT_SCALE]

Options:
    --input_scale INPUT_SCALE, --is INPUT_SCALE
                        Scales values in the main input image before
                        filtering, without scaling the output too

Options:
    -h, --help            Show this help message and exit.
    -m MODEL, --model MODEL
                          Required. Path to an .xml file with a trained model.
    --input_hdr HDR             Required. Path to an HDR image to infer
    --input_albedo ALBEDO       Required. Path to an albedo image to infer
    -d DEVICE, --device DEVICE
                          Optional. Specify the target device to infer on. The
                          demo will look for a suitable plugin for device
                          specified. Default value is CPU
    --no_show             Optional. Don't show output. Cannot be used in GUI mode
    -o OUTPUT, --output OUTPUT
                          Optional. Save output to the file with provided filename.
    -ob OUTPUT_BLOB, --output_blob OUTPUT_BLOB
                          Optional. Name of the output layer of the model.
                          Default is None, in which case the demo will read the
                          output name from the model, assuming there is only 1 output layer

```

For example, to do inference on a CPU with the OpenVINO&trade; toolkit pre-trained `denoise_rt_hdr_alb` model, run the following command:

```sh
python denoise_ray_tracing_demo.py \
  --model <path_to_model>/denoise_rt_hdr_alb.xml \
  --input_hdr data/color.exr \
  --input_albedo data/albedo.exr \
  --output result.exr
```

## Demo Output

The demo uses OpenCV window to display and save the resulting image. The demo reports


* **Latency**: total processing time required to process input data (from preprocessing the data to displaying the results).

## See Also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvino.ai/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/model_tools/README.md)
