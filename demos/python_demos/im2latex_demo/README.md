# Im2LaTeX Python\* Demo

This demo shows how to run im2latex models. Im2latex models allow us to get latex formula markup from the image.

> **NOTE**: Only batch size of 1 is supported.

## How It Works

The demo application expects an im2latex model that is split into two parts. Every model part must be in the Intermediate Representation (IR) format.

First model is Encoder which extracts features from image and prepares first steps of the decoder

* One input is `imgs` for input image 
* Four outputs including:
    * `row_enc_out` extracts features from the image
    * `hidden` and 
    * `context` are intermediate states of the LSTM
    * `init_0` - first state of the encoder

Second model is Decode Step that takes as input:
* `row_enc_out` - extracted images features from the encoder
* `dec_st_c` and
* `dec_st_h` - current states of the LSTM
* `O_t_minus_1` - previous output of the Decode Step (for the first time it is `init_0` of the encoder)
* `tgt` - previous token (for the first time it is `START_TOKEN` )
Second model is being executed until current decoded token is `END_TOKEN` or length of the formula is less then `--max_formula_len` producing one token per each decode step.

As input, the demo application takes a path to a single image file with a command-line argument `-i`.

The demo workflow is the following:

1. The demo application reads image/video frames one by one, crops and pads them to fit into the input image blob of the network (`imgs`). Crop and pad is used to keep size of the font.
2. For each image, encoder extracts features from the image
3. While length of the current formula is less then `--max_formula_len` or current token is not `END_TOKEN` Decode Step produces new tokens.
5. The demo results in the text form into the console or in file if `-o` parameter specified. 

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

## Running

Run the application with the `-h` option to see the following usage message:

```
usage: im2latex_demo.py [-h] -m_encoder ENCODER -m_dec_step DEC_STEP -i INPUT
                        [INPUT ...] [-o OUTPUT_FILE] [-l CPU_EXTENSION]
                        [--vocab_path VOCAB_PATH] --target_shape TARGET_SHAPE
                        [TARGET_SHAPE ...] [--max_formula_len MAX_FORMULA_LEN]
                        [-d DEVICE] [-pf]

Options:
  -h, --help            Show this help message and exit.
  -m_encoder M_ENCODER     Required. Path to an .xml file with a trained encoder
                        part of the model
  -m_dec_step M_DEC_STEP   Required. Path to an .xml file with a trained decoder
                        step part of the model
  -i INPUT [INPUT ...], --input INPUT [INPUT ...]
                        Required. Path to a folder with images or path to an
                        image files
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
                        Optional. Path to file where to store output. If not
                        mentioned, result will be storedin the console.
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        Optional. Required for CPU custom layers. Absolute
                        MKLDNN (CPU)-targeted custom layers. Absolute path to
                        a shared library with the kernels implementations
  --vocab_path VOCAB_PATH
                        Path to vocab file to construct meaningful phrase
  --target_shape TARGET_SHAPE [TARGET_SHAPE ...]
                        Required. Target image shape (height, width). Example:
                        100 500
  --max_formula_len MAX_FORMULA_LEN
                        Optional. Defines maximum length of the formula
                        (number of tokens to decode)
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU,
                        GPU, FPGA, HDDL or MYRIAD is acceptable. Sample will
                        look for a suitable plugin for device specified.
                        Default value is CPU
  -pf, --perf_stats
```

Running the application with an empty list of options yields the short version of the usage message and an error message.

To run the demo, you can use public or pre-trained models. To download the pre-trained models, use the OpenVINO [Model Downloader](../../../tools/downloader/README.md) or go to [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (`*.xml` + `*.bin`) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

To run the demo, please provide paths to the model in the IR format and to an input video or folder with images:
```bash
python im2latex_demo.py \
        -m_encoder <path_to_models>/encoder.xml \
        -m_dec_step <path_to_models>/decode_step.xml \
        --vocab_path <path_to_vocab> \
        --target_shape <heigth width> \
        -i input_image.png
```

## Demo Output

The application outputs recognized formula into the console or into the file.

## See Also
* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
