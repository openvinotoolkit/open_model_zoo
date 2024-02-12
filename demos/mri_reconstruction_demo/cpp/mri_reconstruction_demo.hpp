// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>
#include <iostream>

static const char help_message[] = "Print a usage message.";
static const char input_message[] = "Required. Path to input .npy file with MRI scan data.";
static const char model_message[] = "Required. Path to an .xml file with a trained model.";
static const char target_device_message[] = "Optional. Specify the target device to infer on; CPU or "
                                            "GPU is acceptable (CPU by default).";
static const char pattern_message[] = "Required. Path to sampling mask in .npy format.";
static const char no_show_message[] = "Optional. Disable results visualization.";

DEFINE_bool(h, false, help_message);
DEFINE_string(i, "", input_message);
DEFINE_string(m, "", model_message);
DEFINE_string(d, "CPU", target_device_message);
DEFINE_string(p, "", pattern_message);
DEFINE_bool(no_show, false, no_show_message);

/**
* @brief This function show a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "mri_reconstruction_demo [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                                " << help_message << std::endl;
    std::cout << "    -i \"<path>\"                       " << input_message << std::endl;
    std::cout << "    -p \"<path>\"                       " << pattern_message << std::endl;
    std::cout << "    -m \"<path>\"                       " << model_message << std::endl;
    std::cout << "    -d \"<device>\"                     " << target_device_message << std::endl;
    std::cout << "    --no_show                         " << no_show_message << std::endl;
}
