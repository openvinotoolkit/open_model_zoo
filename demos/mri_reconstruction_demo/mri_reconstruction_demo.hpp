// Copyright (C) 2018-2019 Intel Corporation
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
static const char target_device_message[] = "Optional. Specify the target device to infer on; CPU, "
                                            "GPU, HDDL or MYRIAD is acceptable. For non-CPU targets, "
                                            "HETERO plugin is used with CPU fallbacks to FFT implementation. (CPU by default)";
static const char custom_cpu_library_message[] = "Required. Path to extensions library with FFT implementation.";
static const char pattern_message[] = "Required. Path to sampling mask in .npy format.";

DEFINE_string(l, "", custom_cpu_library_message);
DEFINE_bool(h, false, help_message);
DEFINE_string(i, "", input_message);
DEFINE_string(m, "", model_message);
DEFINE_string(d, "CPU", target_device_message);
DEFINE_string(p, "", pattern_message);

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
    std::cout << "    -l \"<absolute_path>\"            " << custom_cpu_library_message << std::endl;
    std::cout << "    -d \"<device>\"                     " << target_device_message << std::endl;
}
