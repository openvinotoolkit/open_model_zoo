# Imagenet Classification C++ Demo

The demo shows an example of using neural networks for image classification.

You can use the following pre-trained models with the demo:

* `resnet-50`
* `vgg19`
* `googlenet-v4`
* all other classification models (please, check models.lst file)

For more information about the pre-trained models, refer to the [model documentation](../../models/public/index.md).

## How It Works

On the start-up, the application reads command line parameters and loads one network to the Inference Engine for execution. Upon getting an image, it performs inference of classification and places the processed image on grid.

The demo starts in "Testing mode" with fixed grid size. After calculating the average FPS result, it will switch to
normal mode and grid will be readjusted depending on model performance. Bigger grid means higher performance.

The text above each image shows whether the classification was correct: green means right class prediction, red means wrong.

Use "C", "D" and "M" keys to toggle resource motitors.

You can stop the demo by pressing "Esc" button. After that, the average telemetry values would be printed to the console.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

## Running

Running the application with the <code>-h</code> option yields the following usage message:
```sh
./imagenet_classification_demo -h

imagenet_classification_demo [OPTION]
Options:

    -h                        Print a usage message.
    -i "<path>"               Required. Path to a folder with images or path to an image files: a .ubyte file for LeNet and a .bmp file for the other networks.
    -m "<path>"               Required. Path to an .xml file with a trained model.
      -l "<absolute_path>"    Required for CPU custom layers.Absolute path to a shared library with the kernels implementation.
          Or
      -c "<absolute_path>"    Required for GPU custom kernels. Absolute path to the .xml file with kernels description.
    -labels "<path>"          Required. Path to .txt file with imagenet labels.
    -gt "<path>"              Optional. Path to .txt file with image classes.
    -d "<device>"             Optional. Specify the target device to infer on (the list of available devices is shown below). Default value is CPU. Sample will look for a suitable plugin for device specified.
    -b "<integer>"            Optional. Specify batch to infer. Default value is 1.
    -nthreads "<integer>"     Optional. Specify count of threads.
    -nstreams "<integer>"     Optional. Specify count of streams.
    -nireq "<integer>"        Optional. Number of infer requests.
    -nt "<integer>"           Optional. Number of top results. Default value is 5. Must be >= 1.
    -res "<WxH>"              Optional. Set image grid resolution in format WxH. Default value is 1920x1080.
    -no_show                  Optional. Disable showing of processed images.
    -time "<integer>"         Optional. Time in seconds to execute program. Default is -1 (infinite time).
    -u                        Optional. List of monitors to show initially.
```

The number of infer requests can be set by "-nireq" flag.
* small nireq: low FPS (device cores aren't fully loaded) and low latency (images don't wait for long before being sent
for inference).
* nireq = number of cores: high FPS (full device utilization), median latency (images wait for some time). Considered optimal.
* nireq > number of cores: high FPS (same as in the previous case), high latency (images wait for long time).

Running the application with the empty list of options yields an error message.

To run the demo, you can use public or pre-trained models. To download the pre-trained models, use the OpenVINO [Model Downloader](../../tools/downloader/README.md) or go to [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

For example, use the following command line command to run the application:
```sh
./imagenet_classification_demo -m <path_to_classification_model> \
                      -i <path_to_folder_with_images> \
                      -labels <path_to_file_with_list_of_labels> \
                      -gt <path_to_file_with_list_of_classes> \
                      -u CDM
```

## Demo Output

The demo uses OpenCV to display the resulting image grid with classification results presented as a text above images.
After the completion, it prints average telemetry values to the console.

## Required files

If you want to see classification results, you must use "-gt" and "-labels" flags to specify two .txt files
containing lists of classes and labels.

"Classes" file is used for matching image file names with right object classes.
It has the following format:

./ILSVRC2012_val_00000001.JPEG 65
./ILSVRC2012_val_00000002.JPEG 970
./ILSVRC2012_val_00000003.JPEG 230
...

"Labels" file contains the list of human-readable labels, one line for each class.
You can find this file in source code: /open_model_zoo/demos/imagenet_classification_demos/synset_words.txt
It has the following format:

n01440764 tench, Tinca tinca
n01443537 goldfish, Carassius auratus
n01484850 great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias
...

> **NOTE**: On VPU devices (Intel® Movidius™ Neural Compute Stick, Intel® Neural Compute Stick 2, and Intel® Vision Accelerator Design with Intel® Movidius™ VPUs) this demo has been tested on the following Model Downloader available topologies: 
>* `resnet-50`
>* `vgg19`
>* `googlenet-v4`
> Other models may produce unexpected results on these devices.

## See Also
* [Using Open Model Zoo demos](../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../tools/downloader/README.md)