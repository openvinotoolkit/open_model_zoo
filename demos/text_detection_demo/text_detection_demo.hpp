// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>

static const char help_message[] = "Print a usage message.";
static const char input_message[] = "Required. Path to an image or video file, to a text file with paths to images, "
                                    "or to a webcamera device node (for example, /dev/video0).";
static const char text_detection_model_message[] = "Required. Path to the Text Detection model (.xml) file.";
static const char text_recognition_model_message[] = "Required. Path to the Text Recognition model (.xml) file.";
static const char text_recognition_model_symbols_set_message[] = "Optional. Symbol set for the Text Recognition model.";
static const char text_central_image_crop_message[] = "Optional. If it is set, then in case of absence of the Text Detector, "
                                                      "the Text Recognition model takes a central image crop as an input, but not full frame.";
static const char image_width_for_text_detection_model_message[] = "Optional. Input image width for Text Detection model.";
static const char image_height_for_text_detection_model_message[] = "Optional. Input image height for Text Detection model.";
static const char text_recognition_threshold_message[] = "Optional. Specify a recognition confidence threshold. Text detection candidates with "
                                                         "text recognition confidence below specified threshold are rejected.";
static const char pixel_classification_threshold_message[] = "Optional. Specify a confidence threshold for pixel classification. "
                                                             "Pixels with classification confidence below specified threshold are rejected.";
static const char pixel_linkage_threshold_message[] = "Optional. Specify a confidence threshold for pixel linkage. "
                                                      "Pixels with linkage confidence below specified threshold are not linked.";
static const char text_max_rectangles_number_message[] = "Optional. Maximum number of rectangles to recognize. "
                                                         "If it is negative, number of rectangles to recognize is not limited.";
static const char text_detection_target_device_message[] = "Optional. Specify the target device for the Text Detection model to infer on "
                                                           "(the list of available devices is shown below). "
                                                           "The demo will look for a suitable plugin for a specified device. By default, it is CPU.";
static const char text_recognition_target_device_message[] = "Optional. Specify the target device for the Text Recognition model to infer on "
                                                             "(the list of available devices is shown below). "
                                                             "The demo will look for a suitable plugin for a specified device. By default, it is CPU.";
static const char custom_cpu_library_message[] = "Optional. Absolute path to a shared library with the CPU kernels implementation "
                                                 "for custom layers.";
static const char custom_gpu_library_message[] = "Optional. Absolute path to the GPU kernels implementation for custom layers.";
static const char no_show_message[] = "Optional. If it is true, then detected text will not be shown on image frame. By default, it is false.";
static const char raw_output_message[] = "Optional. Output Inference results as raw values.";
static const char input_data_type_message[] = "Required. Input data type: \"image\" (for a single image), "
                                              "\"list\" (for a text file where images paths are listed), "
                                              "\"video\" (for a saved video), "
                                              "\"webcam\" (for a webcamera device). By default, it is \"image\".";
static const char utilization_monitors_message[] = "Optional. List of monitors to show initially.";
static const char decoder_bandwidth_message[] = "Optional. Bandwidth for CTC beam search decoder. Default value is 0, in this case CTC greedy decoder will be used.";

DEFINE_bool(h, false, help_message);
DEFINE_string(i, "", input_message);
DEFINE_string(m_td, "", text_detection_model_message);
DEFINE_string(m_tr, "", text_recognition_model_message);
DEFINE_string(m_tr_ss, "0123456789abcdefghijklmnopqrstuvwxyz", text_recognition_model_symbols_set_message);
DEFINE_bool(cc, false, text_central_image_crop_message);
DEFINE_int32(w_td, 1280, image_width_for_text_detection_model_message);
DEFINE_int32(h_td, 768, image_height_for_text_detection_model_message);
DEFINE_double(thr, 0.2, text_recognition_threshold_message);
DEFINE_double(cls_pixel_thr, 0.8, pixel_classification_threshold_message);
DEFINE_double(link_pixel_thr, 0.8, pixel_linkage_threshold_message);
DEFINE_int32(max_rect_num, -1, text_max_rectangles_number_message);
DEFINE_string(dt, "", input_data_type_message);
DEFINE_string(d_td, "CPU", text_detection_target_device_message);
DEFINE_string(d_tr, "CPU", text_recognition_target_device_message);
DEFINE_string(l, "", custom_cpu_library_message);
DEFINE_string(c, "", custom_gpu_library_message);
DEFINE_bool(no_show, false, no_show_message);
DEFINE_bool(r, false, raw_output_message);
DEFINE_string(u, "", utilization_monitors_message);
DEFINE_uint32(b, 0, decoder_bandwidth_message);

/**
* @brief This function shows a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "text_detection_demo [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                           " << help_message << std::endl;
    std::cout << "    -i \"<path>\"                  " << input_message << std::endl;
    std::cout << "    -m_td \"<path>\"               " << text_detection_model_message << std::endl;
    std::cout << "    -m_tr \"<path>\"               " << text_recognition_model_message << std::endl;
    std::cout << "    -dt \"<input_data_type>\"      " << input_data_type_message << std::endl;
    std::cout << "    -m_tr_ss \"<value>\"           " << text_recognition_model_symbols_set_message << std::endl;
    std::cout << "    -cc                          " << text_central_image_crop_message << std::endl;
    std::cout << "    -w_td \"<value>\"              " << image_width_for_text_detection_model_message << std::endl;
    std::cout << "    -h_td \"<value>\"              " << image_height_for_text_detection_model_message << std::endl;
    std::cout << "    -thr \"<value>\"               " << text_recognition_threshold_message << std::endl;
    std::cout << "    -cls_pixel_thr \"<value>\"     " << pixel_classification_threshold_message << std::endl;
    std::cout << "    -link_pixel_thr \"<value>\"    " << pixel_linkage_threshold_message << std::endl;
    std::cout << "    -max_rect_num \"<value>\"      " << text_max_rectangles_number_message << std::endl;
    std::cout << "    -d_td \"<device>\"             " << text_detection_target_device_message << std::endl;
    std::cout << "    -d_tr \"<device>\"             " << text_recognition_target_device_message << std::endl;
    std::cout << "    -l \"<absolute_path>\"         " << custom_cpu_library_message << std::endl;
    std::cout << "    -c \"<absolute_path>\"         " << custom_gpu_library_message << std::endl;
    std::cout << "    -no_show                     " << no_show_message << std::endl;
    std::cout << "    -r                           " << raw_output_message << std::endl;
    std::cout << "    -u                           " << utilization_monitors_message << std::endl;
    std::cout << "    -b                           " << decoder_bandwidth_message << std::endl;
}
