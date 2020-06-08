# Text Spotting Python* Demo

This demo shows how to run Text Spotting models. Text Spotting models allow us to simultaneously detect and recognize text.

> **NOTE**: Only batch size of 1 is supported.

## How It Works

The demo application expects a text spotting model that is split into three parts. Every model part must be in the Intermediate Representation (IR) format.

First model is Mask-RCNN like text detector with the following constraints:

* Two inputs: `im_data` for input image and `im_info` for meta-information about the image (actual height, width and scale).
* At least five outputs including:
    * `boxes` with absolute bounding box coordinates of the input image
    * `scores` with confidence scores for all bounding boxes
    * `classes` with object class IDs for all bounding boxes
    * `raw_masks` with fixed-size segmentation heat maps for all classes of all bounding boxes
    * `text_features` with text features which are fed to Text Recognition Head further

Second model is Text Recognition Encoder that takes `text_features` as input and produces `encoded text`.

Third model is Text Recognition Decoder that takes `encoded text` from Text Recognition Encoder ,`previous symbol` and `hidden state`. On the first step special `Start Of Sequence (SOS)` symbol and zero `hidden state` are fed to Text Recognition Decoder. The decoder produces `symbols distribution`, `current hidden state` each step until `End Of Sequence (EOS)` symbol is generated.

Examples of valid inputs to specify with a command-line argument `-i` are a path to a video file or a numeric ID of a web camera.

The demo workflow is the following:

1. The demo application reads frames from the provided input, resizes them to fit into the input image blob of the network (`im_data`).
2. The `im_info` input blob passes resulting resolution and scale of a pre-processed image to the network to perform inference of Mask-RCNN-like text detector.
3. The Text Recognition Encoder takes input from the text detector and produces output.
4. The Text Recognition Decoder takes output from the Text Recognition Encoder output as input and produces output.
5. The demo visualizes the resulting text spotting results. Certain command-line options affect the visualization:
    * If you specify `--show_boxes` and `--show_scores` arguments, bounding boxes and confidence scores are also shown.
    * By default, tracking is used to show text instance with the same color throughout the whole video.
      It assumes more or less static scene with instances in two frames being a part of the same track if intersection over union of the masks is greater than the 0.5 threshold. To disable tracking, specify the `--no_track` argument.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

## Running

Run the application with the `-h` option to see the following usage message:

```
usage: text_spotting_demo.py [-h] -m_m "<path>" -m_te "<path>" -m_td "<path>"
                             -i "<path>" [-d "<device>"]
                             [-l "<absolute_path>"] [--delay "<num>"]
                             [-pt "<num>"] [-a ALPHABET]
                             [--trd_input_prev_symbol TRD_INPUT_PREV_SYMBOL]
                             [--trd_input_prev_hidden TRD_INPUT_PREV_HIDDEN]
                             [--trd_input_encoder_outputs TRD_INPUT_ENCODER_OUTPUTS]
                             [--trd_output_symbols_distr TRD_OUTPUT_SYMBOLS_DISTR]
                             [--trd_output_cur_hidden TRD_OUTPUT_CUR_HIDDEN]
                             [--keep_aspect_ratio] [--no_track]
                             [--show_scores] [--show_boxes] [-pc] [-r]
                             [--no_show] [-u UTILIZATION_MONITORS]

Options:
  -h, --help            Show this help message and exit.
  -m_m "<path>", --mask_rcnn_model "<path>"
                        Required. Path to an .xml file with a trained Mask-
                        RCNN model with additional text features output.
  -m_te "<path>", --text_enc_model "<path>"
                        Required. Path to an .xml file with a trained text
                        recognition model (encoder part).
  -m_td "<path>", --text_dec_model "<path>"
                        Required. Path to an .xml file with a trained text
                        recognition model (decoder part).
  -i "<path>"           Required. Input to process.
  -d "<device>", --device "<device>"
                        Optional. Specify the target device to infer on: CPU,
                        GPU, FPGA, HDDL or MYRIAD. The demo will look for a
                        suitable plugin for device specified (by default, it
                        is CPU).
  -l "<absolute_path>", --cpu_extension "<absolute_path>"
                        Required for CPU custom layers. Absolute path to a
                        shared library with the kernels implementation.
  --delay "<num>"       Optional. Interval in milliseconds of waiting for a
                        key to be pressed.
  -pt "<num>", --prob_threshold "<num>"
                        Optional. Probability threshold for detections
                        filtering.
  -a ALPHABET, --alphabet ALPHABET
                        Optional. Alphabet that is used for decoding.
  --trd_input_prev_symbol TRD_INPUT_PREV_SYMBOL
                        Optional. Name of previous symbol input node to text
                        recognition head decoder part.
  --trd_input_prev_hidden TRD_INPUT_PREV_HIDDEN
                        Optional. Name of previous hidden input node to text
                        recognition head decoder part.
  --trd_input_encoder_outputs TRD_INPUT_ENCODER_OUTPUTS
                        Optional. Name of encoder outputs input node to text
                        recognition head decoder part.
  --trd_output_symbols_distr TRD_OUTPUT_SYMBOLS_DISTR
                        Optional. Name of symbols distribution output node
                        from text recognition head decoder part.
  --trd_output_cur_hidden TRD_OUTPUT_CUR_HIDDEN
                        Optional. Name of current hidden output node from text
                        recognition head decoder part.
  --keep_aspect_ratio   Optional. Force image resize to keep aspect ratio.
  --no_track            Optional. Disable tracking.
  --show_scores         Optional. Show detection scores.
  --show_boxes          Optional. Show bounding boxes.
  -pc, --perf_counts    Optional. Report performance counters.
  -r, --raw_output_message
                        Optional. Output inference results raw values.
  --no_show             Optional. Don't show output
  -u UTILIZATION_MONITORS, --utilization_monitors UTILIZATION_MONITORS
                        Optional. List of monitors to show initially.
```

Running the application with an empty list of options yields the short version of the usage message and an error message.

To run the demo, you can use public or pre-trained models. To download the pre-trained models, use the OpenVINO [Model Downloader](../../../tools/downloader/README.md) or go to [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (`*.xml` + `*.bin`) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

To run the demo, please provide paths to the model in the IR format and to an input with images:
```bash
python3 text_spotting_demo.py \
-m_m <path_to_models>/text-spotting-0002-detector.xml \
-m_te <path_to_models>/text-spotting-0002-recognizer-encoder.xml \
-m_td <path_to_models>/text-spotting-0002-recognizer-decoder.xml \
-i 0
```

## Demo Output

The application uses OpenCV to display resulting text instances and current inference performance.

## See Also
* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
