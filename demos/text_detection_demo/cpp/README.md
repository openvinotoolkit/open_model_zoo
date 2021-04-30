# Text Detection C++ Demo

The demo shows an example of using neural networks to detect and recognize printed text rotated at any angle in various environment. You can use the following pre-trained models with the demo:

* `text-detection-0003`, which is a detection network for finding text.
* `text-detection-0004`, which is a lightweight detection network for finding text.
* `horizontal-text-detection-0001`, which is a detection network that works much faster than models above, but it is applicable to finding more or less horizontal text only.
* `text-recognition-0012`, which is a recognition network for recognizing text.
* `text-recognition-0013`, which is a recognition network for recognizing text. You should add option `-tr_pt_first` and specify output layer name via `-tr_o_blb_nm` option for this model (see model [description](../../../models/intel/text-recognition-0013/README.md) for details).
* `text-recognition-resnet-fc`, which is a recognition network for recognizing text. You should add option `-tr_pt_first`.
* `handwritten-score-recognition-0001`, which is a recognition network for recognizing handwritten score marks like `<digit>` or `<digit>.<digit>`.

For more information about the pre-trained models, refer to the [model documentation](../../../models/intel/index.md).

## How It Works

On the start-up, the application reads command line parameters and loads one network to the Inference Engine for execution. Upon getting an image, it performs inference of text detection and prints the result as four points (`x1`, `y1`), (`x2`, `y2`), (`x3`, `y3`), (`x4`, `y4`) for each text bounding box.

If text recognition model is provided, the demo prints recognized text as well.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

## Running

Running the application with the `-h` option yields the following usage message:
```
text_detection_demo [OPTION]
Options:

    -h                           Print a usage message.
    -i                           Required. An input to process. The input must be a single image, a folder of images, video file or camera id.
    -loop                        Optional. Enable reading the input in a loop.
    -o "<path>"                Optional. Name of output to save.
    -limit "<num>"             Optional. Number of frames to store in output. If 0 is set, all frames are stored.
    -m_td "<path>"               Required. Path to the Text Detection model (.xml) file.
    -m_tr "<path>"               Required. Path to the Text Recognition model (.xml) file.
    -dt "<type>"               Optional. Type of the decoder, either 'simple' for SimpleDecoder or 'ctc' for CTC greedy and CTC beam search decoders.
    -m_tr_ss "<value>"           Optional. Symbol set for the Text Recognition model.
    -tr_pt_first                   Optional. Specifies if pad token is the first symbol in the alphabet. Default is false
    -tr_o_blb_nm                   Optional. Name of the output blob of the model which would be used as model output. If not stated, first blob of the model would be used.
    -cc                          Optional. If it is set, then in case of absence of the Text Detector, the Text Recognition model takes a central image crop as an input, but not full frame.
    -w_td "<value>"              Optional. Input image width for Text Detection model.
    -h_td "<value>"              Optional. Input image height for Text Detection model.
    -thr "<value>"               Optional. Specify a recognition confidence threshold. Text detection candidates with text recognition confidence below specified threshold are rejected.
    -cls_pixel_thr "<value>"     Optional. Specify a confidence threshold for pixel classification. Pixels with classification confidence below specified threshold are rejected.
    -link_pixel_thr "<value>"    Optional. Specify a confidence threshold for pixel linkage. Pixels with linkage confidence below specified threshold are not linked.
    -max_rect_num "<value>"      Optional. Maximum number of rectangles to recognize. If it is negative, number of rectangles to recognize is not limited.
    -d_td "<device>"             Optional. Specify the target device for the Text Detection model to infer on (the list of available devices is shown below). The demo will look for a suitable plugin for a specified device. By default, it is CPU.
    -d_tr "<device>"             Optional. Specify the target device for the Text Recognition model to infer on (the list of available devices is shown below). The demo will look for a suitable plugin for a specified device. By default, it is CPU.
    -l "<absolute_path>"         Optional. Absolute path to a shared library with the CPU kernels implementation for custom layers.
    -c "<absolute_path>"         Optional. Absolute path to the GPU kernels implementation for custom layers.
    -no_show                     Optional. If it is true, then detected text will not be shown on image frame. By default, it is false.
    -r                           Optional. Output Inference results as raw values.
    -u                           Optional. List of monitors to show initially.
    -b                           Optional. Bandwidth for CTC beam search decoder. Default value is 0, in this case CTC greedy decoder will be used.
```

Running the application with the empty list of options yields the usage message given above and an error message.

To run the demo, you can use public or pre-trained models. To download the pre-trained models, use the OpenVINO [Model Downloader](../../../tools/downloader/README.md). The list of models supported by the demo is in `<omz_dir>/demos/text_detection_demo/cpp/models.lst`.

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

For example, use the following command line command to run the application:
```sh
./text_detection_demo -m_td <path_to_model>/text-detection-0004.xml \
                      -m_tr <path_to_model>/text-recognition-0013.xml \
                      -i <path_to_image>/sample.jpg \
                      -tr_pt_first \
                      -tr_o_blb_nm "logits"
```

For `text-recognition-resnet-fc` you should use `simple` decoder for `-dt` option. For other models use `ctc` decoder (default decoder).

## Demo Output

The demo uses OpenCV to display the resulting frame with detections rendered as bounding boxes and text.

> **NOTE**: On VPU devices (Intel® Movidius™ Neural Compute Stick, Intel® Neural Compute Stick 2, and Intel® Vision Accelerator Design with Intel® Movidius™ VPUs) this demo is not supported with any of the Model Downloader available topologies. Other models may work incorrectly on these devices as well.

## See Also
* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
