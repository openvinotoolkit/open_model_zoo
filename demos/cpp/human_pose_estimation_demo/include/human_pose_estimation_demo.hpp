// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gflags/gflags.h>
#include <iostream>

static const char help_message[] = "Print a usage message.";
static const char video_message[] = "Required. Path to a video. Default value is \"cam\" to work with camera.";
static const char human_pose_estimation_model_message[] = "Required. Path to the Human Pose Estimation model (.xml) file.";
static const char target_device_message[] = "Optional. Specify the target device for Human Pose Estimation "
                                            "(the list of available devices is shown below). Default value is CPU. "
                                            "Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. "
                                            "The application looks for a suitable plugin for the specified device.";
static const char performance_counter_message[] = "Optional. Enable per-layer performance report.";
static const char no_show_processed_video[] = "Optional. Do not show processed video.";
static const char black_background[] = "Optional. Show black background.";
static const char raw_output_message[] = "Optional. Output inference results as raw values.";
static const char utilization_monitors_message[] = "Optional. List of monitors to show initially.";

DEFINE_bool(h, false, help_message);
DEFINE_string(i, "cam", video_message);
DEFINE_string(m, "", human_pose_estimation_model_message);
DEFINE_string(d, "CPU", target_device_message);
DEFINE_bool(pc, false, performance_counter_message);
DEFINE_bool(no_show, false, no_show_processed_video);
DEFINE_bool(black, false, black_background);
DEFINE_bool(r, false, raw_output_message);
DEFINE_string(u, "", utilization_monitors_message);

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
    std::cout << "    -black                     " << black_background << std::endl;
    std::cout << "    -r                         " << raw_output_message << std::endl;
    std::cout << "    -u                         " << utilization_monitors_message << std::endl;
}
