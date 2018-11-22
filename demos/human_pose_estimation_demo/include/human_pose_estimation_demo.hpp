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

#pragma once

#include <gflags/gflags.h>
#include <iostream>

/// @brief Message for help argument
static const char help_message[] = "Print a usage message.";

/// @brief Message for video argument
static const char video_message[] = "Required. Path to a video. Default value is \"cam\" to work with camera.";

/// @brief Message for model argument
static const char human_pose_estimation_model_message[] = "Required. Path to the Human Pose Estimation model (.xml) file.";

/// @brief Message for assigning Human Pose Estimation inference to device
static const char target_device_message[] = "Optional. Specify the target device for Human Pose Estimation "\
                                            "(CPU, GPU, FPGA or MYRIAD is acceptable). Default value is \"CPU\".";

/// @brief Message for performance counter
static const char performance_counter_message[] = "Optional. Enable per-layer performance report.";

/// @brief Message for not showing processed video
static const char no_show_processed_video[] = "Optional. Do not show processed video.";

/// @brief Message for raw output
static const char raw_output_message[] = "Optional. Output inference results as raw values.";

/// @brief Defines flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// @brief Defines parameter for setting video file <br>
/// It is a required parameter
DEFINE_string(i, "cam", video_message);

/// @brief Defines parameter for human pose estimation model file <br>
/// It is a required parameter
DEFINE_string(m, "", human_pose_estimation_model_message);

/// @brief Defines parameter for the target device to infer on <br>
/// It is an optional parameter
DEFINE_string(d, "CPU", target_device_message);

/// @brief Defines flag for per-layer performance report <br>
/// It is an optional parameter
DEFINE_bool(pc, false, performance_counter_message);

/// @brief Defines flag for disabling processed video showing <br>
/// It is an optional parameter
DEFINE_bool(no_show, false, no_show_processed_video);

/// @brief Defines flag to output raw results <br>
/// It is an optional parameter
DEFINE_bool(r, false, raw_output_message);

/**
* @brief This function shows a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "human_pose_estimation_demo [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                         " << help_message << std::endl;
    std::cout << "    -i \"<path>\"                " << video_message << std::endl;
    std::cout << "    -m \"<path>\"                " << human_pose_estimation_model_message << std::endl;
    std::cout << "    -d \"<device>\"              " << target_device_message << std::endl;
    std::cout << "    -pc                        " << performance_counter_message << std::endl;
    std::cout << "    -no_show                   " << no_show_processed_video << std::endl;
    std::cout << "    -r                         " << raw_output_message << std::endl;
}
