// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>
#include <iostream>

/// @brief message for help argument
static const char help_message[] = "Print a usage message.";

/// @brief message for images argument
static const char image_message[] = "Required. Path to a folder with images or path to an image files: a .ubyte " \
                                    "file for LeNet and a .bmp file for the other networks.";

/// @brief message for model argument
static const char model_message[] = "Required. Path to an .xml file with a trained model.";

/// @brief message for assigning cnn calculation to device
static const char target_device_message[] = "Optional. Specify the target device to infer on (the list of available " \
                                            "devices is shown below). Default value is CPU. " \
                                            "Sample will look for a suitable plugin for device specified.";

/// @brief message for batch size 
static const char batch_size_message[] = "Optional. Specify batch to infer. " \
                                         "Default value is 1.";

/// @brief message for thread num 
static const char num_threads_message[] = "Optional. Specify count of threads.";

/// @brief message for streams num 
static const char num_streams_message[] = "Optional. Specify count of streams.";

/// @brief message for top results number
static const char num_inf_req_message[] = "Optional. Number of infer requests.";

/// @brief message for top results number
static const char delay_message[] = "Optional. Delay between screen updates in milliseconds. " \
                                    "Default value is 1.";

/// @brief Message for setting image grid resolution
static const char image_grid_resolution_message[] = "Optional. Set image grid resolution in format WxH. " \
                                                    "Default value is 1920x1080.";

/// @brief Message for setting image grid cell resolution
static const char image_grid_cell_resolution_message[] = "Optional. Set image grid cell resolution in format WxH. " \
                                                         "Default value is 240x135.";

/// @brief message for top results number
static const char ntop_message[] = "Optional. Number of top results. Default value is 10.";

/// @brief message for clDNN custom kernels desc
static const char custom_cldnn_message[] = "Required for GPU custom kernels. " \
                                           "Absolute path to the .xml file with kernels description.";

/// @brief message for user library argument
static const char custom_cpu_library_message[] = "Required for CPU custom layers." \
                                                 "Absolute path to a shared library with the kernels implementation.";

/// @brief message for plugin messages
static const char plugin_message[] = "Optional. Enables messages from a plugin.";

/// @brief Don't show processed images
static const char no_show_message[] = "Optional. Not showing processed images.";


/// @brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// @brief Define parameter for set image file <br>
/// It is a required parameter
DEFINE_string(i, "", image_message);

/// @brief Define parameter for set model file <br>
/// It is a required parameter
DEFINE_string(m, "", model_message);

/// @brief device the target device to infer on <br>
DEFINE_string(d, "CPU", target_device_message);

/// @brief batch size (default 1)<br>
DEFINE_uint32(b, 1, batch_size_message);

DEFINE_uint32(nthreads, 0, num_threads_message);

DEFINE_string(nstreams, "", num_streams_message);

/// @brief Number of infer request <br>
DEFINE_uint32(nireq, 0, num_inf_req_message);

/// @brief Number of top results <br>
DEFINE_uint32(nt, 10, ntop_message);

/// @brief Delay between screen updates in milliseconds (default 1) <br>
DEFINE_uint32(delay, 1, delay_message);

/// \brief Set image grid resolution in format WxH<br>
/// It is an optional parameter
/// Default is 1920x1080
DEFINE_string(res, "1920x1080", image_grid_resolution_message);

/// \brief Set image grid cell resolution in format WxH<br>
/// It is an optional parameter
/// Default is 240x135
DEFINE_string(cell_res, "240x135", image_grid_cell_resolution_message);

/// @brief Define parameter for clDNN custom kernels path <br>
/// Default is ./lib
DEFINE_string(c, "", custom_cldnn_message);

/// @brief Absolute path to CPU library with user layers <br>
/// It is a optional parameter
DEFINE_string(l, "", custom_cpu_library_message);

/// @brief Enable plugin messages
DEFINE_bool(p_msg, false, plugin_message);

/// @brief Disable showing of processed images
DEFINE_bool(no_show, false, no_show_message);

/**
* @brief This function show a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "imagenet_classification_demo [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                        " << help_message << std::endl;
    std::cout << "    -i \"<path>\"               " << image_message << std::endl;
    std::cout << "    -m \"<path>\"               " << model_message << std::endl;
    std::cout << "      -l \"<absolute_path>\"    " << custom_cpu_library_message << std::endl;
    std::cout << "          Or" << std::endl;
    std::cout << "      -c \"<absolute_path>\"    " << custom_cldnn_message << std::endl;
    std::cout << "    -d \"<device>\"             " << target_device_message << std::endl;
    std::cout << "    -b \"<integer>\"            " << batch_size_message << std::endl;
    std::cout << "    -nthreads \"<integer>\"     " << num_threads_message << std::endl;
    std::cout << "    -nstreams \"<integer>\"     " << num_streams_message << std::endl;
    std::cout << "    -nireq \"<integer>\"        " << num_inf_req_message << std::endl;
    std::cout << "    -nt \"<integer>\"           " << ntop_message << std::endl;
    std::cout << "    -delay \"<integer>\"        " << delay_message << std::endl; 
    std::cout << "    -p_msg                    " << plugin_message << std::endl;
    std::cout << "    -res \"<WxH>\"              " << image_grid_resolution_message << std::endl;
    std::cout << "    -cell_res \"<WxH>\"         " << image_grid_cell_resolution_message << std::endl;
    std::cout << "    -no_show                  " << no_show_message << std::endl;
}
