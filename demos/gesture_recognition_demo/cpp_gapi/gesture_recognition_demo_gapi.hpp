// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <iostream>

#include <gflags/gflags.h>

#include <utils/default_flags.hpp>

DEFINE_INPUT_FLAGS
DEFINE_OUTPUT_FLAGS

static const char help_message[] = "Print a usage message.";
static const char camera_resolution_message[] = "Optional. Set camera resolution in format WxH.";
static const char person_detection_model_message[] =
    "Required. Path to an .xml file with a trained person detector model.";
static const char action_recognition_model_message[] =
    "Required. Path to an .xml file with a trained gesture recognition model.";
static const char target_device_message_d[] =
    "Optional. Target device for Person Detection network. "
    "The demo will look for a suitable plugin for a specified device. Default value is \"CPU\".";
static const char target_device_message_a[] =
    "Optional. Target device for Action Recognition network. "
    "The demo will look for a suitable plugin for a specified device. Default value is \"CPU\".";
static const char thresh_output_message[] =
    "Optional. Threshold for the predicted score of an action. The default value is 0.4.";
static const char class_map_message[] = "Required. Path to a file with gesture classes.";
static const char samples_dir_message[] = "Optional. Path to a .json file that contains paths to samples of gestures.";
static const char no_show_message[] = "Optional. Don't show output.";
static const char utilization_monitors_message[] = "Optional. List of monitors to show initially.";

DEFINE_bool(h, false, help_message);
DEFINE_string(res, "1280x720", camera_resolution_message);
DEFINE_string(m_a, "", action_recognition_model_message);
DEFINE_string(m_d, "", person_detection_model_message);
DEFINE_string(d_a, "CPU", target_device_message_a);
DEFINE_string(d_d, "CPU", target_device_message_d);
DEFINE_string(c, "", class_map_message);
DEFINE_string(s, "", samples_dir_message);
DEFINE_double(t, 0.8, thresh_output_message);
DEFINE_bool(no_show, false, no_show_message);
DEFINE_string(u, "", utilization_monitors_message);

/**
 * \brief This function shows a help message
 */

static void showUsage() {
    std::cout << std::endl;
    std::cout << "gesture_recognition_demo_gapi [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                       " << help_message << std::endl;
    std::cout << "    -i                       " << input_message << std::endl;
    std::cout << "    -loop                    " << loop_message << std::endl;
    std::cout << "    -o \"<path>\"              " << output_message << std::endl;
    std::cout << "    -limit \"<num>\"           " << limit_message << std::endl;
    std::cout << "    -res \"<WxH>\"             " << camera_resolution_message << std::endl;
    std::cout << "    -m_d \"<path>\"            " << person_detection_model_message << std::endl;
    std::cout << "    -m_a \"<path>\"            " << action_recognition_model_message << std::endl;
    std::cout << "    -d_d \"<device>\"          " << target_device_message_d << std::endl;
    std::cout << "    -d_a \"<device>\"          " << target_device_message_a << std::endl;
    std::cout << "    -no_show                 " << no_show_message << std::endl;
    std::cout << "    -c                       " << class_map_message << std::endl;
    std::cout << "    -s                       " << samples_dir_message << std::endl;
    std::cout << "    -t                       " << thresh_output_message << std::endl;
    std::cout << "    -u                       " << utilization_monitors_message << std::endl;
}
