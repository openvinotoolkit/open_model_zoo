/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>
#include <iostream>

#ifdef _WIN32
#include <os/windows/w_dirent.h>
#else
#include <dirent.h>
#endif

/// @brief message for help argument
static const char help_message[] = "Print a usage message.";

/// @brief message for images argument
static const char video_message[] = "Required. Path to video or image files. Default value is \"cam\" to work with cameras.";

/// @brief message for model argument
static const char vehicle_detection_model_message[] = "Required. Path to the Vehicle and License Plate Detection model .xml file.";
static const char vehicle_attribs_model_message[] = "Optional. Path to the Vehicle Attributes model .xml file.";
static const char lpr_model_message[] = "Optional. Path to the License Plate Recognition model .xml file.";

/// @brief message for assigning vehicle detection inference to device
static const char target_device_message[] = "Optional. Specify the target device for Vehicle Detection "\
                                            "(CPU, GPU, FPGA, MYRIAD, or HETERO).";

/// @brief message for assigning vehicle attributes to device
static const char target_device_message_vehicle_attribs[] = "Optional. Specify the target device for Vehicle Attributes "\
                                                            "(CPU, GPU, FPGA, MYRIAD, or HETERO).";

/// @brief message for assigning LPR inference to device
static const char target_device_message_lpr[] = "Optional. Specify the target device for License Plate Recognition "\
                                                "(CPU, GPU, FPGA, MYRIAD, or HETERO).";

/// @brief message for performance counters
static const char performance_counter_message[] = "Optional. Enable per-layer performance statistics.";

/// @brief message for clDNN custom kernels desc
static const char custom_cldnn_message[] = "Optional. For GPU custom kernels, if any. "\
"Absolute path to an .xml file with the kernels description.";

/// @brief message for user library argument
static const char custom_cpu_library_message[] = "Optional. For CPU custom layers, if any. "\
"Absolute path to a shared library with the kernels implementation.";

/// @brief message for probability threshold argument
static const char thresh_output_message[] = "Optional. Probability threshold for vehicle and license plate detections.";

/// @brief message raw output flag
static const char raw_output_message[] = "Optional. Output inference results as raw values.";

/// @brief message no show processed video
static const char no_show_processed_video[] = "Optional. Do not show processed video.";

/// @brief message resizable input flag
static const char input_resizable_message[] = "Optional. Enable resizable input with support of ROI crop and auto resize.";

/// @brief message for number of infer requests
static const char ninfer_request_message[] = "Optional. Number of infer request for pipelined mode (default value is 1).";

/// @brief message for number of camera inputs
static const char num_cameras[] = "Optional. Number of processed cameras (default value is 1) if the input (-i) is specified as camera.";


/// \brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// \brief Define parameter for set image file <br>
/// It is a required parameter
DEFINE_string(i, "cam", video_message);

/// \brief Define parameter for vehicle detection  model file <br>
/// It is a required parameter
DEFINE_string(m, "", vehicle_detection_model_message);

/// \brief Define parameter for vehicle attributes model file <br>
/// It is a required parameter
DEFINE_string(m_va, "", vehicle_attribs_model_message);

/// \brief Define parameter for vehicle detection  model file <br>
/// It is a required parameter
DEFINE_string(m_lpr, "", lpr_model_message);

/// \brief device the target device for vehicle detection infer on <br>
DEFINE_string(d, "CPU", target_device_message);

/// \brief device the target device for age gender detection on <br>
DEFINE_string(d_va, "CPU", target_device_message_vehicle_attribs);

/// \brief device the target device for head pose detection on <br>
DEFINE_string(d_lpr, "CPU", target_device_message_lpr);

/// \brief Enable per-layer performance report
DEFINE_bool(pc, false, performance_counter_message);

/// @brief clDNN custom kernels path <br>
/// Default is ./lib
DEFINE_string(c, "", custom_cldnn_message);

/// @brief Absolute path to CPU library with user layers <br>
/// It is a optional parameter
DEFINE_string(l, "", custom_cpu_library_message);

/// \brief Flag to output raw scoring results<br>
/// It is an optional parameter
DEFINE_bool(r, false, raw_output_message);

/// \brief Flag to output raw scoring results<br>
/// It is an optional parameter
DEFINE_double(t, 0.5, thresh_output_message);

/// \brief Flag to disable processed video showing<br>
/// It is an optional parameter
DEFINE_bool(no_show, false, no_show_processed_video);

/// \brief Enables resizable input<br>
/// It is an optional parameter
DEFINE_bool(auto_resize, false, input_resizable_message);

/// @brief Number of infer requests
/// It is an optional parameter
DEFINE_int32(nireq, 1, ninfer_request_message);

/// \brief Flag to specify number of expected input channels<br>
/// It is an optional parameter
DEFINE_uint32(nc, 1, num_cameras);

/**
* \brief This function show a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "interactive_vehicle_detection [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                         " << help_message << std::endl;
    std::cout << "    -i \"<path1>\" \"<path2>\"     " << video_message << std::endl;
    std::cout << "    -m \"<path>\"                " << vehicle_detection_model_message<< std::endl;
    std::cout << "    -m_va \"<path>\"             " << vehicle_attribs_model_message << std::endl;
    std::cout << "    -m_lpr \"<path>\"            " << lpr_model_message << std::endl;
    std::cout << "      -l \"<absolute_path>\"     " << custom_cpu_library_message << std::endl;
    std::cout << "          Or" << std::endl;
    std::cout << "      -c \"<absolute_path>\"     " << custom_cldnn_message << std::endl;
    std::cout << "    -d \"<device>\"              " << target_device_message << std::endl;
    std::cout << "    -d_va \"<device>\"           " << target_device_message_vehicle_attribs << std::endl;
    std::cout << "    -d_lpr \"<device>\"          " << target_device_message_lpr << std::endl;
    std::cout << "    -pc                        " << performance_counter_message << std::endl;
    std::cout << "    -r                         " << raw_output_message << std::endl;
    std::cout << "    -t                         " << thresh_output_message << std::endl;
    std::cout << "    -no_show                   " << no_show_processed_video << std::endl;
    std::cout << "    -auto_resize               " << input_resizable_message << std::endl;
    std::cout << "    -nireq                     " << ninfer_request_message << std::endl;
    std::cout << "    -nc                        " << num_cameras << std::endl;
}
