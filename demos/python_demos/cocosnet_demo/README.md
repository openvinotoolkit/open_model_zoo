# Cross-domain correspondence network (CoCosNet) Python* Demo

This CoCosNet demo application demonstrates how to work Cross-domain correspondence network, which synthesizes a photo-realistic image from the input in a semantic segmentation mask and exemplar image.

## How It Works

At the start-up the demo application reads command line parameters and loads a network to the Inference Engine plugin. \
To get the result, the demo performs the following steps:

1. Reading input data (semantic segmentation mask of image for translation, exemplar image and mask of exemplar image).
2. Preprocessing for input image and masks.
3. Network inference (correspondence network + generative network).
4. Display the result and save it (optional).

## Running

Running the application with the `-h` option yields the following usage message:

```
python3 cocosnet_demo.py -h
```
The command yields the following usage message:
```
usage: cocosnet_demo.py [-h] -c CORRESPONDENCE_MODEL -g GENERATIVE_MODEL -is
                        INPUT_SEMANTICS -ri REFERENCE_IMAGE -rs
                        REFERENCE_SEMANTICS [-o OUTPUT_DIR] [-l CPU_EXTENSION]
                        [-d DEVICE]

Options:
  -h, --help            Show this help message and exit.
  -c CORRESPONDENCE_MODEL, --correspondence_model CORRESPONDENCE_MODEL
                        Required. Path to an .xml file with a trained
                        correspondence model
  -g GENERATIVE_MODEL, --generative_model GENERATIVE_MODEL
                        Required. Path to an .xml file with a trained
                        generative model
  -is INPUT_SEMANTICS, --input_semantics INPUT_SEMANTICS
                        Required. Path to a folder with semantic images or
                        path to a semantic image
  -ri REFERENCE_IMAGE, --reference_image REFERENCE_IMAGE
                        Required. Path to a folder with reference images or
                        path to a reference image
  -rs REFERENCE_SEMANTICS, --reference_semantics REFERENCE_SEMANTICS
                        Required. Path to a folder with reference semantics or
                        path to a reference semantic
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Path to directory to save the result
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        Optional. Required for CPU custom layers. Absolute
                        MKLDNN (CPU)-targeted custom layers. Absolute path to
                        a shared library with the kernels implementations
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU,
                        GPU, FPGA, HDDL or MYRIAD is acceptable. Sample will
                        look for a suitable plugin for device specified.
                        Default value is CPU

```

Running the application with the empty list of options yields the usage message given above and an error message.

To run the demo, you can use public or pre-trained models. You can download the pre-trained models with the OpenVINO [Model Downloader](../../../tools/downloader/README.md) or from [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

You can use the following command do inference on CPU:
```
    python3 cocosnet_demo.py \
    -c <path_to_corr_model>/Corr.xml \
    -g <path_to_gen_model>/Gen.xml \
    -is <path_to_semantic_mask_of_image>/input_mask.png
    -ri <path_to_exemplar_image>/reference_image.jpg
    -rs <path_to_exemplar_semantic>/reference_mask.png
```

## Demo Output

The application demo uses OpenCV to display the result image. \
Also result can be saved if the `output_dir` is specified.

## See Also

* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
