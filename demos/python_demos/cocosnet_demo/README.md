# Cross-domain correspondence network (CoCosNet) Python* Demo

This CoCosNet demo application demonstrates how to work Cross-domain correspondence network, which synthesizes a photo-realistic image from the input in a semantic segmentation mask and exemplar image.

## How It Works

At the start-up the demo application reads command line parameters and loads a network to the Inference Engine plugin. \
To get the result, the demo performs the following steps:

1. Reading input data (semantic segmentation mask of image for translation, exemplar image and mask of exemplar image).
2. Preprocessing for input image and masks.
3. Network inference (segmentation network (optional) + correspondence network + generative network).
4. Save results to folder.

## Running

Running the application with the `-h` option yields the following usage message:

```
python3 cocosnet_demo.py -h
```

The command yields the following usage message:

```
usage: cocosnet_demo.py [-h] -c CORRESPONDENCE_MODEL -g GENERATIVE_MODEL
                        [-s SEGMENTATION_MODEL] [-ii INPUT_IMAGES]
                        [-is INPUT_SEMANTICS] -ri REFERENCE_IMAGES
                        [-rs REFERENCE_SEMANTICS] [-o OUTPUT_DIR] [-d DEVICE]

Options:
  -h, --help            Show this help message and exit.
  -c CORRESPONDENCE_MODEL, --correspondence_model CORRESPONDENCE_MODEL
                        Required. Path to an .xml file with a trained
                        correspondence model
  -g GENERATIVE_MODEL, --generative_model GENERATIVE_MODEL
                        Required. Path to an .xml file with a trained
                        generative model
  -s SEGMENTATION_MODEL, --segmentation_model SEGMENTATION_MODEL
                        Optional. Path to an .xml file with a trained
                        semantic segmentation model
  -ii INPUT_IMAGES, --input_images INPUT_IMAGES
                        Optional. Path to a folder with input images or path
                        to a input image
  -is INPUT_SEMANTICS, --input_semantics INPUT_SEMANTICS
                        Optional. Path to a folder with semantic images or
                        path to a semantic image
  -ri REFERENCE_IMAGES, --reference_images REFERENCE_IMAGES
                        Required. Path to a folder with reference images or
                        path to a reference image
  -rs REFERENCE_SEMANTICS, --reference_semantics REFERENCE_SEMANTICS
                        Optional. Path to a folder with reference semantics
                        or path to a reference semantic
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Optional. Path to directory to save the results
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU,
                        GPU, FPGA, HDDL or MYRIAD is acceptable. Default
                        value is CPU
```

Running the application with the empty list of options yields the usage message given above and an error message.

To run the demo, you can use public or pre-trained models. You can download the pre-trained models with the OpenVINO [Model Downloader](../../../tools/downloader/README.md) or from [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

There are two ways to use this demo:

1. To use only correspondence and generative models (only CoCosNet). \
   In this case user have to set 3 inputs to the model (2 masks and 1 image).
   You can use the following command run demo on CPU:

   ```
   python3 cocosnet_demo.py \
       -d CPU \
       -c <path_to_corr_model>/Corr.xml \
       -g <path_to_gen_model>/Gen.xml \
       -is <path_to_semantic_mask_of_image>/input_mask.png \
       -ri <path_to_exemplar_image>/reference_image.jpg \
       -rs <path_to_exemplar_semantic>/reference_mask.png
   ```

   > **NOTE**: For segmentation masks you should use mask (with shape: [height x width]) that specifies class for each pixel. Number of classes is 151 (from ADE20k), where '0' - nothing class.

2. To use the segmentation model in addition to CoCosNet. You should use only models trained on ADE20k dataset.     Example: [hrnet-v2-c1-segmentation](../../../models/public/hrnet-v2-c1-segmentation/hrnet-v2-c1-segmentation.md).
   In this case user have to set input image (.jpg) and reference image (.jpg) without any masks.
   Segmentation masks will be generated via segmentation model.

   You can use the following command run demo on CPU:

   ```
   python3 cocosnet_demo.py \
       -d CPU \
       -c <path_to_corr_model>/Corr.xml \
       -g <path_to_gen_model>/Gen.xml \
       -s <path_to_seg_model>/Seg.xml \
       -ii <path_to_input_image>/input_image.jpg \
       -ri <path_to_exemplar_image>/reference_image.jpg
   ```

## Demo Output

The results of the demo processing are saved to a folder that is specified by the parameter `output_dir`.

## See Also

* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
