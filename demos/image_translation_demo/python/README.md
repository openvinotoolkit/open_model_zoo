# Image Translation Demo

This demo application demonstrates an example of using neural networks to synthesize a photo-realistic image based on an exemplar image.

## How It Works

On startup the demo application reads command line parameters and loads a network to the Inference Engine plugin. To get the result, the demo performs the following steps:

1. Reading input data (semantic segmentation mask of image for translation, exemplar image and mask of exemplar image).
2. Preprocessing for input image and masks.
3. Network inference (segmentation network (optional) + translation network).
4. Save results to folder.

## Preparing to Run

For demo input image or video files you may refer to [Media Files Available for Demos](../../README.md#Media-Files-Available-for-Demos).
The list of models supported by the demo is in `<omz_dir>/demos/image_translation_demo/python/models.lst` file.
This file can be used as a parameter for [Model Downloader](../../../tools/downloader/README.md) and Converter to download and, if necessary, convert models to OpenVINO Inference Engine format (\*.xml + \*.bin).

An example of using the Model Downloader:

```sh
python3 <omz_dir>/tools/downloader/downloader.py --list models.lst
```

An example of using the Model Converter:

```sh
python3 <omz_dir>/tools/downloader/converter.py --list models.lst
```

### Supported Models

* cocosnet
* hrnet-v2-c1-segmentation

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.

## Running

Running the application with the `-h` option yields the following usage message:

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
                        GPU, HDDL or MYRIAD is acceptable. Default
                        value is CPU
```

Running the application with the empty list of options yields the usage message given above and an error message.

There are two ways to use this demo:

1. Run with segmentation model in addition to translation model. You should use only models trained on ADE20k dataset. Example: [hrnet-v2-c1-segmentation](../../../models/public/hrnet-v2-c1-segmentation/README.md).
   In this case only input and reference images are required without any masks. Segmentation masks will be generated via segmentation model.

   You can use the following command to run demo on CPU using cocosnet and hrnet-v2-c1-segmentation models:

   ```sh
   python3 image_translation_demo.py \
       -d CPU \
       -m_trn <path_to_translation_model>/cocosnet.xml \
       -m_seg <path_to_segmentation_model>/hrnet-v2-c1-segmentation.xml \
       -ii <path_to_input_image>/input_image.jpg \
       -ri <path_to_exemplar_image>/reference_image.jpg \
       -o <output_dir>
   ```

2. Run with only translation model.
   You can use the following command to run demo on CPU using cocosnet as translation model:

   ```sh
   python3 image_translation_demo.py \
       -d CPU \
       -m_trn <path_to_translation_model>/cocosnet.xml \
       -is <path_to_semantic_mask_of_image>/input_mask.png \
       -ri <path_to_exemplar_image>/reference_image.jpg \
       -rs <path_to_exemplar_semantic>/reference_mask.png \
       -o <output_dir>
   ```

   > **NOTE**: For segmentation masks you should use mask (with shape: [height x width]) that specifies class for each pixel. Number of classes is 151 (from ADE20k), where '0' - background class.

## Demo Output

The results of the demo processing are saved to a folder that is specified by the parameter `output_dir`.

## See Also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
