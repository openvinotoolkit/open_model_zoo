// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>

#ifdef _WIN32
#include <os/windows/w_dirent.h>
#else
#include <dirent.h>
#endif

/// @brief Message for help argument
static const char help_message[] = "Print a usage message.";

/// @brief Message for image argument
static const char image_message[] = "Required. Path to an image file.";

/// @brief Message for model argument
static const char model_message[] = "Required. Path to the Text Detection model (.xml) file.";

/// @brief Message for target device argument
static const char target_device_message[] = "Optional. Specify the target device to infer on: CPU, GPU, FPGA, or MYRIAD. "
                                            "The demo will look for a suitable plugin for a specified device.";

/// @brief Message for user library argument
static const char custom_cpu_library_message[] = "Optional. Absolute path to a shared library with the CPU kernels implementation "
                                                 "for custom layers.";

/// @brief Message for user library argument
static const char custom_gpu_library_message[] = "Optional. Absolute path to the GPU kernels implementation for custom layers.";

/// @brief Message for user no_show argument
static const char no_show_message[] = "Optional. If it is true, then detected text will not be shown on image frame. By default, it is false.";

/// @brief Message raw output flag
static const char raw_output_message[] = "Optional. Output Inference results as raw values.";

/// @brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// @brief Define parameter for setting image file <br>
/// It is a required parameter
DEFINE_string(i, "", image_message);

/// @brief Define parameter for text detection model file <br>
/// It is a required parameter
DEFINE_string(m, "", model_message);

/// @brief Define the target device to infer on <br>
DEFINE_string(d, "CPU", target_device_message);

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
    std::cout << "    -i \"<path>\"                  " << image_message << std::endl;
    std::cout << "    -m \"<path>\"                  " << model_message << std::endl;
    std::cout << "    -d \"<device>\"                " << target_device_message << std::endl;
    std::cout << "    -l \"<absolute_path>\"         " << custom_cpu_library_message << std::endl;
    std::cout << "    -c \"<absolute_path>\"         " << custom_gpu_library_message << std::endl;
    std::cout << "    -no_show                     " << no_show_message << std::endl;
    std::cout << "    -r                           " << raw_output_message << std::endl;
}
