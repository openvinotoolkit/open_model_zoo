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
static const char gaze_estimation_model_message[] =
    "Required. Path to an .xml file with a trained Gaze Estimation model.";
static const char face_detection_model_message[] =
    "Required. Path to an .xml file with a trained Face Detection model.";
static const char head_pose_model_message[] =
    "Required. Path to an .xml file with a trained Head Pose Estimation model.";
static const char facial_landmarks_model_message[] =
    "Required. Path to an .xml file with a trained Facial Landmarks Estimation model.";
static const char eye_state_model_message[] =
    "Required. Path to an .xml file with a trained Open/Closed Eye Estimation model.";
static const char target_device_message[] =
    "Optional. Target device for Gaze Estimation network (the list of available devices is shown below). "
    "The demo will look for a suitable plugin for a specified device. Default value is \"CPU\".";
static const char target_device_message_fd[] =
    "Optional. Target device for Face Detection network (the list of available devices is shown below). "
    "The demo will look for a suitable plugin for a specified device. Default value is \"CPU\".";
static const char target_device_message_hp[] =
    "Optional. Target device for Head Pose Estimation network (the list of available devices is shown below). "
    "The demo will look for a suitable plugin for a specified device. Default value is \"CPU\".";
static const char target_device_message_lm[] =
    "Optional. Target device for Facial Landmarks Estimation network "
    "(the list of available devices is shown below). Use \"-d HETERO:<comma-separated_devices_list>\" format to "
    "specify HETERO plugin. "
    "The demo will look for a suitable plugin for a specified device. Default value is \"CPU\".";
static const char target_device_message_es[] =
    "Optional. Target device for Open/Closed Eye network "
    "(the list of available devices is shown below). Use \"-d HETERO:<comma-separated_devices_list>\" format to "
    "specify HETERO plugin. "
    "The demo will look for a suitable plugin for a specified device. Default value is \"CPU\".";
static const char thresh_output_message[] =
    "Optional. Probability threshold for Face Detector. The default value is 0.5.";
static const char raw_output_message[] = "Optional. Output inference results as raw values.";
static const char fd_reshape_message[] = "Optional. Reshape Face Detector network so that its input resolution has the "
                                         "same aspect ratio as the input frame.";
static const char no_show_message[] = "Optional. Don't show output.";
static const char utilization_monitors_message[] = "Optional. List of monitors to show initially.";

DEFINE_bool(h, false, help_message);
DEFINE_string(res, "1280x720", camera_resolution_message);
DEFINE_string(m, "", gaze_estimation_model_message);
DEFINE_string(m_fd, "", face_detection_model_message);
DEFINE_string(m_hp, "", head_pose_model_message);
DEFINE_string(m_lm, "", facial_landmarks_model_message);
DEFINE_string(m_es, "", eye_state_model_message);
DEFINE_string(d, "CPU", target_device_message);
DEFINE_string(d_fd, "CPU", target_device_message_fd);
DEFINE_string(d_hp, "CPU", target_device_message_hp);
DEFINE_string(d_lm, "CPU", target_device_message_lm);
DEFINE_string(d_es, "CPU", target_device_message_es);
DEFINE_bool(fd_reshape, false, fd_reshape_message);
DEFINE_bool(r, false, raw_output_message);
DEFINE_double(t, 0.5, thresh_output_message);
DEFINE_bool(no_show, false, no_show_message);
DEFINE_string(u, "", utilization_monitors_message);

/**
 * \brief This function shows a help message
 */

static void showUsage() {
    std::cout << std::endl;
    std::cout << "gaze_estimation_demo_gapi [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                       " << help_message << std::endl;
    std::cout << "    -i                       " << input_message << std::endl;
    std::cout << "    -loop                    " << loop_message << std::endl;
    std::cout << "    -o \"<path>\"              " << output_message << std::endl;
    std::cout << "    -limit \"<num>\"           " << limit_message << std::endl;
    std::cout << "    -res \"<WxH>\"             " << camera_resolution_message << std::endl;
    std::cout << "    -m \"<path>\"              " << gaze_estimation_model_message << std::endl;
    std::cout << "    -m_fd \"<path>\"           " << face_detection_model_message << std::endl;
    std::cout << "    -m_hp \"<path>\"           " << head_pose_model_message << std::endl;
    std::cout << "    -m_lm \"<path>\"           " << facial_landmarks_model_message << std::endl;
    std::cout << "    -m_es \"<path>\"           " << eye_state_model_message << std::endl;
    std::cout << "    -d \"<device>\"            " << target_device_message << std::endl;
    std::cout << "    -d_fd \"<device>\"         " << target_device_message_fd << std::endl;
    std::cout << "    -d_hp \"<device>\"         " << target_device_message_hp << std::endl;
    std::cout << "    -d_lm \"<device>\"         " << target_device_message_lm << std::endl;
    std::cout << "    -d_es \"<device>\"         " << target_device_message_es << std::endl;
    std::cout << "    -fd_reshape              " << fd_reshape_message << std::endl;
    std::cout << "    -no_show                 " << no_show_message << std::endl;
    std::cout << "    -r                       " << raw_output_message << std::endl;
    std::cout << "    -t                       " << thresh_output_message << std::endl;
    std::cout << "    -u                       " << utilization_monitors_message << std::endl;
}
