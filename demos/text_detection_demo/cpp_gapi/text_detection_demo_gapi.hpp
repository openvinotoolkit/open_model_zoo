// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>

#include <utils/default_flags.hpp>

DEFINE_INPUT_FLAGS
DEFINE_OUTPUT_FLAGS

static const char helpMsg[] = "Print a usage message.";
static const char tdModelMsg[] =
    "Required. Path to the Text Detection model (.xml) file.";
static const char trModelMsg[] =
    "Required. Path to the Text Recognition model (.xml) file.";
static const char trDecoderMsg[] =
    "Optional. Type of the decoder, either 'simple' for SimpleDecoder or 'ctc' for CTC greedy and"
    " CTC beam search decoders. Default is 'ctc'";
static const char trSymbolsSetMsg[] =
    "Optional. String or vocabulary file with symbol set for the Text Recognition model.";
static const char trLowercaseMsg[] =
    "Optional. Set this flag to convert recognized text to lowercase";
static const char trEncoderOutHiddenMsg[] =
    "Optional. Name of the text recognition model encoder output hidden blob";
static const char trDecoderOutHiddenMsg[] =
    "Optional. Name of the text recognition model decoder output hidden blob";
static const char trDecoderInHidden[] =
    "Optional. Name of the text recognition model decoder input hidden blob";
static const char trFeaturesMsg[] =
    "Optional. Name of the text recognition model features blob";
static const char trDecoderInMsg[] =
    "Optional. Name of the text recognition model decoder input blob (prev. decoded symbol)";
static const char trDecoderOutMsg[] =
    "Optional. Name of the text recognition model decoder output blob "
    "(probability distribution over tokens)";
static const char trOutputMsg[] =
    "Optional. Name of the output blob of the model which would be used as model output. "
    "If not stated, first blob of the model would be used.";
static const char trPadFirstMsg[] =
    "Optional. Specifies if pad token is the first symbol in the alphabet. Default is false";
static const char textCCMsg[] =
    "Optional. If it is set, then in case of absence of the Text Detector, the Text Recognition"
    " model takes a central image crop as an input, but not full frame.";
static const char tdInputWidthMsg[] =
    "Optional. Input image width for Text Detection model.";
static const char tdInputHeightMsg[] =
    "Optional. Input image height for Text Detection model.";
static const char trThresholdMsg[] =
    "Optional. Specify a recognition confidence threshold. Text detection candidates with"
    " text recognition confidence below specified threshold are rejected.";
static const char tdPixelClsThresholdMsg[] =
    "Optional. Specify a confidence threshold for pixel classification. Pixels with classification"
    " confidence below specified threshold are rejected.";
static const char tdPixelLinkThresholdMsg[] =
    "Optional. Specify a confidence threshold for pixel linkage. Pixels with linkage confidence"
    " below specified threshold are not linked.";
static const char trMaxRectNumMsg[] =
    "Optional. Maximum number of rectangles to recognize. If it is negative, number of rectangles"
    " to recognize is not limited.";
static const char tdDeviceMsg[] =
    "Optional. Specify the target device for the Text Detection model to infer on"
    " (the list of available devices is shown below). The demo will look for a suitable plugin"
    " for a specified device. By default, it is CPU.";
static const char trDeviceMsg[] =
    "Optional. Specify the target device for the Text Recognition model to infer on"
    " (the list of available devices is shown below). The demo will look for a suitable plugin"
    " for a specified device. By default, it is CPU.";
static const char noShowMsg[] =
    "Optional. If it is true, then detected text will not be shown on image frame."
    " By default, it is false.";
static const char outRawMsg[] = "Optional. Output Inference results as raw values.";
static const char inDataTypeMsg[] =
    "Required. Input data type: \"image\" (for a single image),"
                              " \"list\" (for a text file where images paths are listed),"
                              " \"video\" (for a saved video),"
                              " \"webcam\" (for a webcamera device)."
    " By default, it is \"image\".";
static const char utilMonitorsMsg[] = "Optional. List of monitors to show initially.";
static const char decoderBandwidthMsg[] =
    "Optional. Bandwidth for CTC beam search decoder."
    " Default value is 0, in this case CTC greedy decoder will be used.";
static const char decoderStartIndexMsg[] =
    "Optional. Start index for Simple decoder. Default value is 0.";
static const char padSymbolMsg[] = "Optional. Pad symbol. Default value is '#'.";

DEFINE_bool(h, false, helpMsg);
DEFINE_string(m_td, "", tdModelMsg);
DEFINE_string(m_tr, "", trModelMsg);
DEFINE_string(dt, "ctc", trDecoderMsg);
DEFINE_string(m_tr_ss, "0123456789abcdefghijklmnopqrstuvwxyz",
              trSymbolsSetMsg);
DEFINE_bool(tr_pt_first, false, trPadFirstMsg);
DEFINE_bool(lower, false, trLowercaseMsg);
DEFINE_string(out_enc_hidden_name, "decoder_hidden", trEncoderOutHiddenMsg);
DEFINE_string(out_dec_hidden_name, "decoder_hidden", trDecoderOutHiddenMsg);
DEFINE_string(in_dec_hidden_name, "hidden", trDecoderInHidden);
DEFINE_string(features_name, "features", trFeaturesMsg);
DEFINE_string(in_dec_symbol_name, "decoder_input", trDecoderInMsg);
DEFINE_string(out_dec_symbol_name, "decoder_output", trDecoderOutMsg);
DEFINE_string(tr_o_blb_nm, "", trOutputMsg);
DEFINE_bool(cc, false, textCCMsg);
DEFINE_int32(w_td, 0, tdInputWidthMsg);
DEFINE_int32(h_td, 0, tdInputHeightMsg);
DEFINE_double(thr, 0.2, trThresholdMsg);
DEFINE_double(cls_pixel_thr, 0.8, tdPixelClsThresholdMsg);
DEFINE_double(link_pixel_thr, 0.8, tdPixelLinkThresholdMsg);
DEFINE_int32(max_rect_num, -1, trMaxRectNumMsg);
DEFINE_string(d_td, "CPU", tdDeviceMsg);
DEFINE_string(d_tr, "CPU", trDeviceMsg);
DEFINE_bool(no_show, false, noShowMsg);
DEFINE_bool(r, false, outRawMsg);
DEFINE_string(u, "", utilMonitorsMsg);
DEFINE_uint32(b, 0, decoderBandwidthMsg);
DEFINE_uint32(start_index, 0, decoderStartIndexMsg);
DEFINE_string(pad, "#", padSymbolMsg);

/**
* @brief This function shows a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "text_detection_demo_gapi [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                                 " << helpMsg << std::endl;
    std::cout << "    -i                                 " << input_message << std::endl;
    std::cout << "    -loop                              " << loop_message << std::endl;
    std::cout << "    -o \"<path>\"                      " << output_message << std::endl;
    std::cout << "    -limit \"<num>\"                   " << limit_message << std::endl;
    std::cout << "    -m_td \"<path>\"                   " << tdModelMsg << std::endl;
    std::cout << "    -m_tr \"<path>\"                   " << trModelMsg << std::endl;
    std::cout << "    -w_td \"<value>\"                  " << tdInputWidthMsg << std::endl;
    std::cout << "    -h_td \"<value>\"                  " << tdInputHeightMsg << std::endl;
    std::cout << "    -d_td \"<device>\"                 " << tdDeviceMsg << std::endl;
    std::cout << "    -d_tr \"<device>\"                 " << trDeviceMsg << std::endl;
    std::cout << "    -dt \"<type>\"                     " << trDecoderMsg << std::endl;
    std::cout << "    -m_tr_ss \"<value>\" or \"<path>\" " << trSymbolsSetMsg << std::endl;
    std::cout << "    -tr_pt_first                       " << trPadFirstMsg << std::endl;
    std::cout << "    -lower                             " << trLowercaseMsg << std::endl;
    std::cout << "    -out_enc_hidden_name \"<value>\"   " << trEncoderOutHiddenMsg << std::endl;
    std::cout << "    -out_dec_hidden_name \"<value>\"   " << trDecoderOutHiddenMsg << std::endl;
    std::cout << "    -in_dec_hidden_name \"<value>\"    " << trDecoderInHidden << std::endl;
    std::cout << "    -features_name \"<value>\"         " << trFeaturesMsg << std::endl;
    std::cout << "    -in_dec_symbol_name \"<value>\"    " << trDecoderInMsg << std::endl;
    std::cout << "    -out_dec_symbol_name \"<value>\"   " << trDecoderOutMsg << std::endl;
    std::cout << "    -tr_o_blb_nm \"<value>\"           " << trOutputMsg << std::endl;
    std::cout << "    -cc                                " << textCCMsg << std::endl;
    std::cout << "    -thr \"<value>\"                   " << trThresholdMsg << std::endl;
    std::cout << "    -cls_pixel_thr \"<value>\"         " << tdPixelClsThresholdMsg << std::endl;
    std::cout << "    -link_pixel_thr \"<value>\"        " << tdPixelLinkThresholdMsg << std::endl;
    std::cout << "    -max_rect_num \"<value>\"          " << trMaxRectNumMsg << std::endl;
    std::cout << "    -no_show                           " << noShowMsg << std::endl;
    std::cout << "    -r                                 " << outRawMsg << std::endl;
    std::cout << "    -u                                 " << utilMonitorsMsg << std::endl;
    std::cout << "    -b                                 " << decoderBandwidthMsg << std::endl;
    std::cout << "    -start_index                       " << decoderStartIndexMsg << std::endl;
    std::cout << "    -pad                               " << padSymbolMsg << std::endl;
}
