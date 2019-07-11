// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

/// @brief Message for help argument
static const char help_message[] = "Print a usage message.";

/// @brief Message for images argument
static const char video_message[] = "Optional. Path to a video file. Default value is \"cam\" to work with camera.";

/// @brief message for model argument
static const char gaze_estimation_model_message[] = "Required. Path to an .xml file with a trained Gaze Estimation model.";
static const char face_detection_model_message[] = "Required. Path to an .xml file with a trained Face Detection model.";
static const char head_pose_model_message[] = "Required. Path to an .xml file with a trained Head Pose Estimation model.";
static const char facial_landmarks_model_message[] = "Required. Path to an .xml file with a trained Facial Landmarks Estimation model.";

/// @brief Message for plugin argument
static const char plugin_message[] = "Plugin name. For example, CPU. If this parameter is specified, " \
"the demo will look for this plugin only.";

/// @brief Message for assigning gaze calculation to device
static const char target_device_message[] = "Optional. Target device for Gaze Estimation network (the list of available devices is shown below). " \
"Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. " \
"The demo will look for a suitable plugin for a specified device. Default value is \"CPU\".";

/// @brief Message for assigning face detection calculation to device
static const char target_device_message_fd[] = "Optional. Target device for Face Detection network (the list of available devices is shown below). " \
"Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. " \
"The demo will look for a suitable plugin for a specified device. Default value is \"CPU\".";

/// @brief Message for assigning head pose calculation to device
static const char target_device_message_hp[] = "Optional. Target device for Head Pose Estimation network (the list of available devices is shown below). " \
"Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. " \
"The demo will look for a suitable plugin for a specified device. Default value is \"CPU\".";

/// @brief Message for assigning facial landmarks calculation to device
static const char target_device_message_lm[] = "Optional. Target device for Facial Landmarks Estimation network " \
"(the list of available devices is shown below). Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. " \
"The demo will look for a suitable plugin for device specified. Default value is \"CPU\".";

/// @brief Message for assigning setting resolution of camera
static const char camera_resolution_message[] = "Optional. Set camera resolution in format WxH.";

/// @brief Message for performance counters
static const char performance_counter_message[] = "Optional. Enable per-layer performance report.";

/// @brief Message for probability threshold argument
static const char thresh_output_message[] = "Optional. Probability threshold for Face Detector. The default value is 0.5.";

/// @brief Message raw output flag
static const char raw_output_message[] = "Optional. Output inference results as raw values.";

/// @brief Message for enabling Face Detector network reshape
static const char fd_reshape_message[] = "Optional. Reshape Face Detector network so that its input resolution has the same aspect ratio as the input frame.";

/// @brief Message do not show processed video
static const char no_show_processed_video[] = "Optional. Do not show processed video.";

/// \brief Define flag for showing help message<br>
DEFINE_bool(h, false, help_message);

/// \brief Define parameter for set image file<br>
/// It is a required parameter
DEFINE_string(i, "cam", video_message);

/// \brief Define parameter for Gaze Estimation model file<br>
/// It is a required parameter
DEFINE_string(m, "", gaze_estimation_model_message);

/// \brief Define parameter for Face Detection model file<br>
/// It is a required parameter
DEFINE_string(m_fd, "", face_detection_model_message);

/// \brief Define parameter for Head Pose Estimation model file<br>
/// It is a required parameter
DEFINE_string(m_hp, "", head_pose_model_message);

/// \brief Define parameter for Facial Landmarks Estimation model file<br>
/// It is an optional parameter
DEFINE_string(m_lm, "", facial_landmarks_model_message);

/// \brief target device for Gaze Estimation network<br>
DEFINE_string(d, "CPU", target_device_message);

/// \brief Define parameter for target device for Face Detection network<br>
DEFINE_string(d_fd, "CPU", target_device_message_fd);

/// \brief Define parameter for target device for Head Pose Estimation network<br>
DEFINE_string(d_hp, "CPU", target_device_message_hp);

/// \brief Define parameter for target device for Facial Landmarks Estimation network<br>
DEFINE_string(d_lm, "CPU", target_device_message_lm);

/// \brief Define parameter camera resolution<br>
/// It is an optional parameter
DEFINE_string(res, "", camera_resolution_message);

/// \brief Define parameter to enable face detector network reshape<br>
/// It is an optional parameter
DEFINE_bool(fd_reshape, false, fd_reshape_message);

/// \brief Define parameter to enable per-layer performance report<br>
/// It is an optional parameter
DEFINE_bool(pc, false, performance_counter_message);

/// \brief Define a flag to output raw scoring results<br>
/// It is an optional parameter
DEFINE_bool(r, false, raw_output_message);

/// \brief Define a parameter for probability threshold for detections<br>
/// It is an optional parameter
DEFINE_double(t, 0.5, thresh_output_message);

/// \brief Define a flag to disable showing processed video<br>
/// It is an optional parameter
DEFINE_bool(no_show, false, no_show_processed_video);

/**
* \brief This function shows a help message
*/

static void showUsage() {
    std::cout << std::endl;
    std::cout << "gaze_estimation_demo [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                       " << help_message << std::endl;
    std::cout << "    -i \"<path>\"              " << video_message << std::endl;
    std::cout << "    -m \"<path>\"              " << gaze_estimation_model_message << std::endl;
    std::cout << "    -m_fd \"<path>\"           " << face_detection_model_message << std::endl;
    std::cout << "    -m_hp \"<path>\"           " << head_pose_model_message << std::endl;
    std::cout << "    -m_lm \"<path>\"           " << facial_landmarks_model_message << std::endl;
    std::cout << "    -d \"<device>\"            " << target_device_message << std::endl;
    std::cout << "    -d_fd \"<device>\"         " << target_device_message_fd << std::endl;
    std::cout << "    -d_hp \"<device>\"         " << target_device_message_hp << std::endl;
    std::cout << "    -d_lm \"<device>\"         " << target_device_message_lm << std::endl;
    std::cout << "    -res \"<WxH>\"             " << camera_resolution_message << std::endl;
    std::cout << "    -fd_reshape              " << fd_reshape_message << std::endl;
    std::cout << "    -no_show                 " << no_show_processed_video << std::endl;
    std::cout << "    -pc                      " << performance_counter_message << std::endl;
    std::cout << "    -r                       " << raw_output_message << std::endl;
    std::cout << "    -t                       " << thresh_output_message << std::endl;
}
