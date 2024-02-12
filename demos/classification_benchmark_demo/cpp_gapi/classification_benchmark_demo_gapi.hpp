// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <iostream>

#include <gflags/gflags.h>

#include <utils_gapi/backend_description.hpp>
#include <utils/args_helper.hpp>

static const char help_message[] = "Print a usage message.";
static const char image_message[] = "Required. Path to a folder with images or path to an image file.";
static const char model_message[] = "Required. Path to an .xml file with a trained model.";
static const char labels_message[] = "Required. Path to .txt file with labels.";
static const char gt_message[] = "Optional. Path to ground truth .txt file.";
static const char target_device_message[] = "Optional. Specify the target device to infer on (the list of available "
                                            "devices is shown below). Default value is CPU. "
                                            "The demo will look for a suitable plugin for device specified.";
static const char num_threads_message[] = "Optional. Specify count of threads.";
static const char num_streams_message[] = "Optional. Specify count of streams.";
static const char num_inf_req_message[] = "Optional. Number of infer requests.";
static const char image_grid_resolution_message[] = "Optional. Set image grid resolution in format WxH. "
                                                    "Default value is 1280x720.";
static const char ntop_message[] = "Optional. Number of top results. Default value is 5. Must be >= 1.";
static const char no_show_message[] = "Optional. Disable showing of processed images.";
static const char execution_time_message[] = "Optional. Time in seconds to execute program. "
                                             "Default is -1 (infinite time).";
static const char utilization_monitors_message[] = "Optional. List of monitors to show initially.";
static const std::string backend_message_str("Optional. Specify an inference backend. The list of available backends: " +
                                             merge(getSupportedInferenceBackends(), ",") + ". Default value is IE. See README.md for details");
static const char *backend_message = backend_message_str.c_str();
static const char mean_values_message[] =
    "Optional. Normalize input by subtracting the mean values per channel. Example: \"255.0 255.0 255.0\"";
static const char scale_values_message[] = "Optional. Divide input by scale values per channel. Division is applied "
                                           "after mean values subtraction. Example: \"255.0 255.0 255.0\"";

DEFINE_bool(h, false, help_message);
DEFINE_string(i, "", image_message);
DEFINE_string(m, "", model_message);
DEFINE_string(labels, "", labels_message);
DEFINE_string(gt, "", gt_message);
DEFINE_string(d, "CPU", target_device_message);
DEFINE_uint32(nthreads, 0, num_threads_message);
DEFINE_string(nstreams, "", num_streams_message);
DEFINE_uint32(nireq, 1, num_inf_req_message);
DEFINE_uint32(nt, 5, ntop_message);
DEFINE_string(res, "1280x720", image_grid_resolution_message);
DEFINE_bool(no_show, false, no_show_message);
DEFINE_uint32(time, std::numeric_limits<gflags::uint32>::max(), execution_time_message);
DEFINE_string(u, "", utilization_monitors_message);
DEFINE_string(backend, "OV", backend_message);
DEFINE_string(mean_values, "", mean_values_message);
DEFINE_string(scale_values, "", scale_values_message);

/**
 * \brief This function shows a help message
 */

static void showUsage() {
    std::cout << std::endl;
    std::cout << "classification_benchmark_demo [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                        " << help_message << std::endl;
    std::cout << "    -i \"<path>\"               " << image_message << std::endl;
    std::cout << "    -m \"<path>\"               " << model_message << std::endl;
    std::cout << "    -labels \"<path>\"          " << labels_message << std::endl;
    std::cout << "    -gt \"<path>\"              " << gt_message << std::endl;
    std::cout << "    -d \"<device>\"             " << target_device_message << std::endl;
    std::cout << "    -nthreads \"<integer>\"     " << num_threads_message << std::endl;
    std::cout << "    -nstreams \"<integer>\"     " << num_streams_message << std::endl;
    std::cout << "    -nireq \"<integer>\"        " << num_inf_req_message << std::endl;
    std::cout << "    -nt \"<integer>\"           " << ntop_message << std::endl;
    std::cout << "    -res \"<WxH>\"              " << image_grid_resolution_message << std::endl;
    std::cout << "    -no_show                  " << no_show_message << std::endl;
    std::cout << "    -time \"<integer>\"         " << execution_time_message << std::endl;
    std::cout << "    -u                        " << utilization_monitors_message << std::endl;
    std::cout << "    -backend                  " << backend_message << std::endl;
    std::cout << "    -mean_values              " << mean_values_message << std::endl;
    std::cout << "    -scale_values             " << scale_values_message << std::endl;
}
