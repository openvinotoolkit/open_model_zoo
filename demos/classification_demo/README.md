# Classification C++ Demo

The demo shows an example of using neural networks for image classification.

You can use the following pre-trained models with the demo:

* `alexnet`
* `resnet-50`
* `vgg19`
* all other classification models (please, check [models.lst](./models.lst) file)

For more information about the pre-trained models, refer to the [model documentation](../../models/public/index.md).

## How It Works

On the start-up, the application reads command line parameters and loads a classification network to the Inference Engine for execution. It might take some time for demo to read all input images. Then the demo performs inference to classify the images and places them on grid.

The demo starts in "Testing mode" with fixed grid size. After calculating the average FPS result, it will switch to normal mode and grid will be readjusted depending on model performance. Bigger grid means higher performance.

The text above each image shows whether the classification was correct: green means correct class prediction, red means wrong.

You can stop the demo by pressing "Esc" or "Q" button. After that, the average metrics values will be printed to the console.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

## Running

Running the application with the <code>-h</code> option yields the following usage message:
```sh
./classification_demo -h

classification_demo [OPTION]
Options:

    -h                        Print a usage message.
    -i "<path>"               Required. Path to a folder with images or path to an image file.
    -m "<path>"               Required. Path to an .xml file with a trained model.
      -l "<absolute_path>"    Required for CPU custom layers.Absolute path to a shared library with the kernels implementation.
          Or
      -c "<absolute_path>"    Required for GPU custom kernels. Absolute path to the .xml file with kernels description.
    -labels "<path>"          Required. Path to .txt file with labels.
    -gt "<path>"              Optional. Path to ground truth .txt file.
    -d "<device>"             Optional. Specify the target device to infer on (the list of available devices is shown below). Default value is CPU. Sample will look for a suitable plugin for device specified.
    -b "<integer>"            Optional. Specify batch to infer. Default value is 1. The following models: googlenet-v4-tf, resnet18-xnor-binary-onnx-0001, resnet50-binary-0001, support only default batch size.
    -nthreads "<integer>"     Optional. Specify count of threads.
    -nstreams "<integer>"     Optional. Specify count of streams.
    -nireq "<integer>"        Optional. Number of infer requests.
    -nt "<integer>"           Optional. Number of top results. Default value is 5. Must be >= 1.
    -res "<WxH>"              Optional. Set image grid resolution in format WxH. Default value is 1280x720.
    -no_show                  Optional. Disable showing of processed images.
    -time "<integer>"         Optional. Time in seconds to execute program. Default is -1 (infinite time).
    -u                        Optional. List of monitors to show initially.
```

The number of `InferRequest`s is specified by -nireq flag. Each `InferRequest` acts as a "buffer": it waits in queue before being filled with images and sent for inference, then after the inference completes, it waits in queue until its results are processed. Increasing the number of `InferRequest`s usually increases performance, because in that case multiple `InferRequest`s can be processed simultaneously if the device supports parallelization. However, big number of `InferRequest`s increases latency because each image still needs to wait in queue.

For higher FPS, using `-nireq` which slightly exceeds `-nstreams` value summed over all used devices is recommended.

Running the application with the empty list of options yields an error message.

To run the demo, you can use public or pre-trained models. To download the pre-trained models, use the OpenVINO [Model Downloader](../../tools/downloader/README.md) or go to [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

For example, use the following command line command to run the application:
```sh
./classification_demo -m <path_to_classification_model> \
                      -i <path_to_folder_with_images> \
                      -labels <path_to_file_with_list_of_labels> \
                      -gt <path_to_ground_truth_data_file> \
                      -u CDM
```

## Demo Output

The demo uses OpenCV to display the resulting image grid with classification results presented as a text above images. After the completion, it prints average metrics values to the console.

## Required files

If you want to see classification results, you must use "-gt" and "-labels" flags to specify two .txt files containing lists of classes and labels.

"Ground truth" file is used for matching image file names with correct object classes.

It has the following format:

```sh
./ILSVRC2012_val_00000001.JPEG 65
./ILSVRC2012_val_00000002.JPEG 970
./ILSVRC2012_val_00000003.JPEG 230
...
```

Class index values must be in range from 0 to 1000. If you want to use "other" class, which is supported only by a small subset of models, specify it with -1 index.

["Labels" file](./synset_words.txt) contains the list of human-readable labels, one line for each class.

## See Also
* [Using Open Model Zoo demos](../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../tools/downloader/README.md)