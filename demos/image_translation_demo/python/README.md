# Image Translation Demo

This demo application demonstrates an example of using neural networks to synthesize a photo-realistic image based on exemplar image. You can use the following models with the demo:

* `cocosnet`
* `hrnet-v2-c1-segmentation`

## How It Works

At the start-up the demo application reads command line parameters and loads a network to the Inference Engine plugin. To get the result, the demo performs the following steps:

1. Reading input data (semantic segmentation mask of image for translation, exemplar image and mask of exemplar image).
2. Preprocessing for input image and masks.
3. Network inference (segmentation network (optional) + translation network).
4. Save results to folder.

## Running

Running the application with the `-h` option yields the following usage message:

```
python3 cocosnet_demo.py -h
```

The command yields the following usage message:

```
usage: image_translation_demo.py [-h] -m_trn TRANSLATION_MODEL
                                 [-m_seg SEGMENTATION_MODEL] [-ii INPUT_IMAGES]
                                 [-is INPUT_SEMANTICS] -ri REFERENCE_IMAGES
                                 [-rs REFERENCE_SEMANTICS] -o OUTPUT_DIR
                                 [-d DEVICE]

Options:
  -h, --help            Show this help message and exit.
  -m_trn TRANSLATION_MODEL, --translation_model TRANSLATION_MODEL
                        Required. Path to an .xml file with a trained
                        translation model
  -m_seg SEGMENTATION_MODEL, --segmentation_model SEGMENTATION_MODEL
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
                        Required. Path to a folder where output files will be
                        saved
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU,
                        GPU, FPGA, HDDL or MYRIAD is acceptable. Default
                        value is CPU

```

Running the application with the empty list of options yields the usage message given above and an error message.

To run the demo, you can use public or pre-trained models. You can download the pre-trained models with the OpenVINO [Model Downloader](../../../tools/downloader/README.md). The list of models supported by the demo is in [models.lst](./models.lst).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

There are two ways to use this demo:

1. Run with segmentation model in addition to translation model. You should use only models trained on ADE20k dataset.     Example: [hrnet-v2-c1-segmentation](../../../models/public/hrnet-v2-c1-segmentation/README.md).
   In this case only input and reference images are required without any masks.
   Segmentation masks will be generated via segmentation model.

   You can use the following command to run demo on CPU using cocosnet and hrnet-v2-c1-segmentation models:

   ```
   python3 image_translation_demo.py \
       -d CPU \
       -m_trn <path_to_translation_model>/cocosnet.xml \
       -m_seg <path_to_segmentation_model>/hrnet-v2-c1-segmentation.xml \
       -ii <path_to_input_image>/input_image.jpg \
       -ri <path_to_exemplar_image>/reference_image.jpg
   ```

2. Run with only translation model.
   You can use the following command to run demo on CPU using cocosnet as translation model:

   ```
   python3 image_translation_demo.py \
       -d CPU \
       -m_trn <path_to_translation_model>/cocosnet.xml \
       -is <path_to_semantic_mask_of_image>/input_mask.png \
       -ri <path_to_exemplar_image>/reference_image.jpg \
       -rs <path_to_exemplar_semantic>/reference_mask.png
   ```

   > **NOTE**: For segmentation masks you should use mask (with shape: [height x width]) that specifies class for each pixel. Number of classes is 151 (from ADE20k), where '0' - background class.

## Demo Output

The results of the demo processing are saved to a folder that is specified by the parameter `output_dir`.

## See Also

* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
