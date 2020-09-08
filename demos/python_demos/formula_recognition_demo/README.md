# Formula Recognition Python\* Demo

This demo shows how to run im2latex models. Im2latex models allow to get a latex formula markup from the image.

> **NOTE**: Only batch size of 1 is supported.

## How It Works

The demo application expects an im2latex model that is split into two parts. Every model part must be in the Intermediate Representation (IR) format.

The First model is Encoder which extracts features from an image and prepares first steps of the decoder.

* One input is `imgs` for input image 
* Four outputs are:
    * Row encoding out (`row_enc_out`) extracts features from the image
    * `hidden` and 
    * `context` are intermediate states of the LSTM
    * `init_0` - first state of the encoder

Second model is Decoder that takes as input:
* `row_enc_out` - extracted images features from the encoder
* Decoding state context (`dec_st_c`) and
* Decoding state hidden (`dec_st_h`) - current states of the LSTM
* `output_prev` - previous output of the Decode Step (for the first time it is `init_0` of the encoder)
* Target (`tgt`) - previous token (for the first time it is `START_TOKEN` )
Second model is being executed until current decoded token is `END_TOKEN` or length of the formula is less then `--max_formula_len` producing one token per each decode step.

As input, the demo application takes a path to a folder with images or a path to a single image file with a command-line argument `-i`.

The demo workflow is the following:

1. The demo application reads a single image or iterates over all images in the given folder, then crops or resizes and inputs to fit into the input image blob of the network (`imgs`). Crop and pad is used to keep size of the font.
2. For each image, encoder extracts features from the image
3. While length of the current formula is less then `--max_formula_len` or current token is not `END_TOKEN` Decode Step produces new tokens.
5. The demo prints the decoded text into the console or in a file if `-o` parameter specified. 

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

The demo has two preprocessing types: Crop and Pad to target shape and Resize and pad to target shape. Two preprocessing types are used for two different datasets as model trained with concrete font size, so if one wants to run the model on inputs with bigger font size (e.g. if input is photographed in 12Mpx, while model trained to imitate scans in ~3Mpx) they should first resize the input to make font size like in train set. Example of the target font size:
![](./sample.png)

## Running

Run the application with the `-h` option to see the following usage message:

```
usage: formula_recognition_demo.py [-h] -m_encoder M_ENCODER -m_decoder M_DECODER -i
                        INPUT [-o OUTPUT_FILE] --vocab_path VOCAB_PATH
                        [--max_formula_len MAX_FORMULA_LEN] [-d DEVICE]
                        [--preprocessing_type {crop,resize}] [-pc]
                        [--imgs_layer IMGS_LAYER]
                        [--row_enc_out_layer ROW_ENC_OUT_LAYER]
                        [--hidden_layer HIDDEN_LAYER]
                        [--context_layer CONTEXT_LAYER]
                        [--init_0_layer INIT_0_LAYER]
                        [--dec_st_c_layer DEC_ST_C_LAYER]
                        [--dec_st_h_layer DEC_ST_H_LAYER]
                        [--dec_st_c_t_layer DEC_ST_C_T_LAYER]
                        [--dec_st_h_t_layer DEC_ST_H_T_LAYER]
                        [--output_layer OUTPUT_LAYER]
                        [--output_prev_layer OUTPUT_PREV_LAYER]
                        [--logit_layer LOGIT_LAYER] [--tgt_layer TGT_LAYER]

Options:
  -h, --help            Show this help message and exit.
  -m_encoder M_ENCODER  Required. Path to an .xml file with a trained encoder
                        part of the model
  -m_decoder M_DECODER  Required. Path to an .xml file with a trained decoder
                        part of the model
  -i INPUT, --input INPUT
                        Required. Path to a folder with images or path to an
                        image files
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
                        Optional. Path to file where to store output. If not
                        mentioned, result will be storedin the console.
  --vocab_path VOCAB_PATH
                        Required. Path to vocab file to construct meaningful
                        phrase
  --max_formula_len MAX_FORMULA_LEN
                        Optional. Defines maximum length of the formula
                        (number of tokens to decode)
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU,
                        GPU, FPGA, HDDL or MYRIAD is acceptable. Sample will
                        look for a suitable plugin for device specified.
                        Default value is CPU
  --preprocessing_type {crop,resize}
                        Optional. Type of the preprocessing
  -pc, --perf_counts
  --imgs_layer IMGS_LAYER
                        Optional. Encoder input key for images. See README for
                        details.
  --row_enc_out_layer ROW_ENC_OUT_LAYER
                        Optional. Encoder output key for row_enc_out. See
                        README for details.
  --hidden_layer HIDDEN_LAYER
                        Optional. Encoder output key for hidden. See README
                        for details.
  --context_layer CONTEXT_LAYER
                        Optional. Encoder output key for context. See README
                        for details.
  --init_0_layer INIT_0_LAYER
                        Optional. Encoder output key for init_0. See README
                        for details.
  --dec_st_c_layer DEC_ST_C_LAYER
                        Optional. Decoder input key for dec_st_c. See README
                        for details.
  --dec_st_h_layer DEC_ST_H_LAYER
                        Optional. Decoder input key for dec_st_h. See README
                        for details.
  --dec_st_c_t_layer DEC_ST_C_T_LAYER
                        Optional. Decoder output key for dec_st_c_t. See
                        README for details.
  --dec_st_h_t_layer DEC_ST_H_T_LAYER
                        Optional. Decoder output key for dec_st_h_t. See
                        README for details.
  --output_layer OUTPUT_LAYER
                        Optional. Decoder output key for output. See README
                        for details.
  --output_prev_layer OUTPUT_PREV_LAYER
                        Optional. Decoder input key for output_prev. See
                        README for details.
  --logit_layer LOGIT_LAYER
                        Optional. Decoder output key for logit. See README for
                        details.
  --tgt_layer TGT_LAYER
                        Optional. Decoder input key for tgt. See README for
                        details.
```

Running the application with an empty list of options yields the short version of the usage message and an error message.

To run the demo, you can use public or pre-trained models. To download the pre-trained models, use the OpenVINO [Model Downloader](../../../tools/downloader/README.md) or go to [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (`*.xml` + `*.bin`) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

To run the demo, please provide paths to the model in the IR format and to an input video or folder with images:
```bash
python formula_recognition_demo.py \
        -m_encoder <path_to_models>/encoder.xml \
        -m_decoder <path_to_models>/decode_step.xml \
        --vocab_path <path_to_vocab> \
        --preprocessing <preproc type> \
        -i input_image.png
```

## Demo Output

The application outputs recognized formula into the console or into the file.

## See Also
* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
