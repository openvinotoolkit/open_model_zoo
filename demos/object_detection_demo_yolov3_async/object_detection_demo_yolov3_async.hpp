// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>
#include <iostream>

/// @brief Message for help argument
static const char help_message[] = "Print a usage message.";

/// @brief Message for images argument
static const char video_message[] = "Required. Path to a video file (specify \"cam\" to work with camera).";

/// @brief Message for model argument
static const char model_message[] = "Required. Path to an .xml file with a trained model.";

/// @brief Message for assigning cnn calculation to device
static const char target_device_message[] = "Optional. Specify a target device to infer on (the list of available devices is shown below). " \
"Default value is CPU. The demo will look for a suitable plugin for the specified device";

/// @brief Message for performance counters
static const char performance_counter_message[] = "Optional. Enable per-layer performance report.";

/// @brief Message for clDNN custom kernels desc
static const char custom_cldnn_message[] = "Optional. Required for GPU custom kernels. "\
"Absolute path to the .xml file with the kernels description.";

/// @brief Message for user library argument
static const char custom_cpu_library_message[] = "Optional. Required for CPU custom layers. " \
"Absolute path to a shared library with the layers implementation.";

/// @brief Message for probability threshold argument
static const char thresh_output_message[] = "Optional. Probability threshold for detections.";

/// @brief Message for probability threshold argument
static const char iou_thresh_output_message[] = "Optional. Filtering intersection over union threshold for overlapping boxes.";

/// @brief Message raw output flag
static const char raw_output_message[] = "Optional. Output inference results raw values showing.";

/// @brief Message resizable input flag
static const char input_resizable_message[] = "Optional. Enable resizable input with support of ROI crop and auto resize.";

/// @brief Message do not show processed video
static const char no_show_processed_video[] = "Optional. Do not show processed video.";

/// \brief Defines flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// \brief Define parameter for set video file of reading from camera <br>
/// It is a required parameter
DEFINE_string(i, "", video_message);

/// \brief Defines parameter for setting path to a model file <br>
/// It is a required parameter
DEFINE_string(m, "", model_message);

/// \brief Defines the target device to infer on <br>
DEFINE_string(d, "CPU", target_device_message);

/// \brief Enables per-layer performance report
DEFINE_bool(pc, false, performance_counter_message);

/// @brief Defines GPU custom kernels path <br>
/// Default is ./lib
DEFINE_string(c, "", custom_cldnn_message);

/// @brief Defines absolute path to CPU library with user layers <br>
/// It is a optional parameter
DEFINE_string(l, "", custom_cpu_library_message);

/// \brief Defines flag to output raw scoring results<br>
/// It is an optional parameter
DEFINE_bool(r, false, raw_output_message);

/// \brief Defines value of confidence threshold <br>
/// It is an optional parameter
DEFINE_double(t, 0.5, thresh_output_message);

/// \brief Defines value of intersection over union threshold<br>
/// It is an optional parameter
DEFINE_double(iou_t, 0.4, iou_thresh_output_message);

/// \brief Enables resizable input<br>
/// It is an optional parameter
DEFINE_bool(auto_resize, false, input_resizable_message);

/// \brief Define a flag to disable showing processed video<br>
/// It is an optional parameter
DEFINE_bool(no_show, false, no_show_processed_video);

/**
* \brief This function shows a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "object_detection_demo_yolov3_async [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                        " << help_message << std::endl;
    std::cout << "    -i \"<path>\"               " << video_message << std::endl;
    std::cout << "    -m \"<path>\"               " << model_message << std::endl;
    std::cout << "      -l \"<absolute_path>\"    " << custom_cpu_library_message << std::endl;
    std::cout << "          Or" << std::endl;
    std::cout << "      -c \"<absolute_path>\"    " << custom_cldnn_message << std::endl;
    std::cout << "    -d \"<device>\"             " << target_device_message << std::endl;
    std::cout << "    -pc                       " << performance_counter_message << std::endl;
    std::cout << "    -r                        " << raw_output_message << std::endl;
    std::cout << "    -t                        " << thresh_output_message << std::endl;
    std::cout << "    -iou_t                    " << iou_thresh_output_message << std::endl;
    std::cout << "    -auto_resize              " << input_resizable_message << std::endl;
    std::cout << "    -no_show                  " << no_show_processed_video << std::endl;
}
