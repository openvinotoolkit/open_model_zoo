// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>

/// @brief Message for help argument
static const char help_message[] = "Print a usage message.";

/// @brief Message for input path argument
static const char input_message[] = "Required. Path to an image or video file, to a text file with paths to images, "
                                    "or to a webcamera device node (for example, /dev/video0).";

/// @brief Message for text detection model argument
static const char text_detection_model_message[] = "Required. Path to the Text Detection model (.xml) file.";

/// @brief Message for text recognition model argument
static const char text_recognition_model_message[] = "Required. Path to the Text Recognition model (.xml) file.";

/// @brief Message for text recognition model symbols set argument
static const char text_recognition_model_symbols_set_message[] = "Optional. Symbol set for the Text Recognition model.";

/// @brief Message for central image crop argument
static const char text_central_image_crop_message[] = "Optional. If it is set, then in case of absence of the Text Detector, "
                                                      "the Text Reconition model takes a central image crop as an input, but not full frame.";

/// @brief Message for input image width for text detection model argument
static const char image_width_for_text_detection_model_message[] = "Optional. Input image width for Text Detection model.";

/// @brief Message for input image height for text detection model argument
static const char image_height_for_text_detection_model_message[] = "Optional. Input image height for Text Detection model.";

/// @brief Message for text recognition threshold argument
static const char text_recognition_threshold_message[] = "Optional. Specify a recognition confidence threshold. Text detection candidates with "
                                                         "text recognition confidence below specified threshold are rejected.";

/// @brief Message for pixel classification threshold argument
static const char pixel_classification_threshold_message[] = "Optional. Specify a confidence threshold for pixel classification. "
                                                             "Pixels with classification confidence below specified threshold are rejected.";

/// @brief Message for pixel linkage threshold argument
static const char pixel_linkage_threshold_message[] = "Optional. Specify a confidence threshold for pixel linkage. "
                                                      "Pixels with linkage confidence below specified threshold are not linked.";

/// @brief Message for max rectangles number argument
static const char text_max_rectangles_number_message[] = "Optional. Maximum number of rectangles to recognize. "
                                                         "If it is negative, number of rectangles to recognize is not limited.";

/// @brief Message for text detection target device argument
static const char text_detection_target_device_message[] = "Optional. Specify the target device for the Text Detection model to infer on "
                                                           "(the list of available devices is shown below). "
                                                           "The demo will look for a suitable plugin for a specified device. By default, it is CPU.";

/// @brief Message for text recognition target device argument
static const char text_recognition_target_device_message[] = "Optional. Specify the target device for the Text Recognition model to infer on "
                                                             "(the list of available devices is shown below). "
                                                             "The demo will look for a suitable plugin for a specified device. By default, it is CPU.";

/// @brief Message for user library argument
static const char custom_cpu_library_message[] = "Optional. Absolute path to a shared library with the CPU kernels implementation "
                                                 "for custom layers.";

/// @brief Message for user library argument
static const char custom_gpu_library_message[] = "Optional. Absolute path to the GPU kernels implementation for custom layers.";

/// @brief Message for user no_show argument
static const char no_show_message[] = "Optional. If it is true, then detected text will not be shown on image frame. By default, it is false.";

/// @brief Message raw output flag
static const char raw_output_message[] = "Optional. Output Inference results as raw values.";

/// @brief Message for input data type argument
static const char input_data_type_message[] = "Required. Input data type: \"image\" (for a single image), "
                                              "\"list\" (for a text file where images paths are listed), "
                                              "\"video\" (for a saved video), "
                                              "\"webcam\" (for a webcamera device). By default, it is \"image\".";

/// @brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// @brief Define parameter for setting input path <br>
/// It is a required parameter
DEFINE_string(i, "", input_message);

/// @brief Define parameter for text detection model file <br>
/// It is a required parameter
DEFINE_string(m_td, "", text_detection_model_message);

/// @brief Define parameter for text recognition model file <br>
/// It is a required parameter
DEFINE_string(m_tr, "", text_recognition_model_message);

/// @brief Define parameter for text recognition model symbols set <br>
/// It is a optional parameter
DEFINE_string(m_tr_ss, "0123456789abcdefghijklmnopqrstuvwxyz", text_recognition_model_symbols_set_message);

/// @brief Define parameter for central image crop. <br>
/// It is a optional parameter
DEFINE_bool(cc, false, text_central_image_crop_message);

/// @brief Define parameter for input image width for text detection model <br>
/// It is a optional parameter
DEFINE_int32(w_td, 1280, image_width_for_text_detection_model_message);

/// @brief Define parameter for input image height for text detection model <br>
/// It is a optional parameter
DEFINE_int32(h_td, 768, image_height_for_text_detection_model_message);

/// @brief Define parameter for text recognition threshold <br>
/// It is a optional parameter
DEFINE_double(thr, 0.2, text_recognition_threshold_message);

/// @brief Define parameter for pixel classification threshold <br>
/// It is a optional parameter
DEFINE_double(cls_pixel_thr, 0.8, pixel_classification_threshold_message);

/// @brief Define parameter for pixel linking threshold <br>
/// It is a optional parameter
DEFINE_double(link_pixel_thr, 0.8, pixel_linkage_threshold_message);

/// @brief Define parameter for maximum number of rectangles to recognize. If it is negative number of rectangles to recognize is not limited. <br>
/// It is a optional parameter
DEFINE_int32(max_rect_num, -1, text_max_rectangles_number_message);

/// @brief Define parameter for input data type ("image", "list", "video", "webcam"). <br>
/// It is a required parameter
DEFINE_string(dt, "", input_data_type_message);

/// @brief Define the target device for text detection model to infer on <br>
DEFINE_string(d_td, "CPU", text_detection_target_device_message);

/// @brief Define the target device for text recognition model to infer on <br>
DEFINE_string(d_tr, "CPU", text_recognition_target_device_message);

/// @brief Define parameter for asolute path to a shared library with the CPU kernels implementation for custom layers. <br>
/// It is a optional parameter
DEFINE_string(l, "", custom_cpu_library_message);

/// @brief Define parameter for asolute path to the GPU kernels implementation for custom layers. <br>
/// It is a optional parameter
DEFINE_string(c, "", custom_gpu_library_message);

/// @brief Define a flag to not show detected text on image frame. By default, it is false. <br>
/// It is an optional parameter
DEFINE_bool(no_show, false, no_show_message);

/// @brief Flag to output raw pipeline results<br>
/// It is an optional parameter
DEFINE_bool(r, false, raw_output_message);

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
}
