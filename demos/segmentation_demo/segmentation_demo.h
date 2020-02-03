// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>
#include <iostream>

static const char help_message[] = "Print a usage message.";
static const char input_message[] = "Required. Input to process.";
static const char model_message[] = "Required. Path to an .xml file with a trained model.";
static const char target_device_message[] = "Optional. Specify the target device to infer on (the list of available devices is shown below). "
                                            "Default value is CPU. Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. "
                                            "The demo will look for a suitable plugin for a specified device.";
static const char custom_cldnn_message[] = "Required for GPU custom kernels. "
                                           "Absolute path to the .xml file with the kernels descriptions.";
static const char custom_cpu_library_message[] = "Required for CPU custom layers. "
                                                 "Absolute path to a shared library with the kernels implementations.";
static const char config_message[] = "Path to the configuration file. Default vaelue: \"config\".";
static const char delay_message[] = "Optional. Default is 1. Interval in milliseconds of waiting for a key to be "
                                    "pressed. For a negative value the demo loads a model, opens an input and "
                                    "exits.";
static const char no_show_message[] = "Optional. Do not visualize inference results.";
static const char utilization_monitors_message[] = "Optional. List of monitors to show initially.";

DEFINE_string(c, "", custom_cldnn_message);
DEFINE_string(l, "", custom_cpu_library_message);
DEFINE_bool(h, false, help_message);
DEFINE_string(i, "", input_message);
DEFINE_string(m, "", model_message);
DEFINE_string(d, "CPU", target_device_message);
DEFINE_string(config, "", config_message);
DEFINE_int32(delay, 1, delay_message);
DEFINE_bool(no_show, false, no_show_message);
DEFINE_string(u, "", utilization_monitors_message);

static void showUsage() {
    std::cout << std::endl;
    std::cout << "segmentation_demo [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                        " << help_message << std::endl;
    std::cout << "    -i \"<path>\"               " << input_message << std::endl;
    std::cout << "    -m \"<path>\"               " << model_message << std::endl;
    std::cout << "      -l \"<absolute_path>\"    " << custom_cpu_library_message << std::endl;
    std::cout << "          Or" << std::endl;
    std::cout << "      -c \"<absolute_path>\"    " << custom_cldnn_message << std::endl;
    std::cout << "    -d \"<device>\"             " << target_device_message << std::endl;
    std::cout << "    -delay                    " << delay_message << std::endl;
    std::cout << "    -no_show                  " << no_show_message << std::endl;
    std::cout << "    -u                        " << utilization_monitors_message << std::endl;
}
