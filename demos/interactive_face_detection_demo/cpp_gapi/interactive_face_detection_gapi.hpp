// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include <gflags/gflags.h>
#include <utils/default_flags.hpp>

DEFINE_OUTPUT_FLAGS

static const char help_message[] = "Print a usage message";
static const char input_video_message[] = "Required. Path to a video file (specify \"cam\" to work with camera).";
static const char face_detection_model_message[] = "Required. Path to an .xml file with a trained Face Detection model.";
static const char age_gender_model_message[] = "Optional. Path to an .xml file with a trained Age/Gender Recognition model.";
static const char head_pose_model_message[] = "Optional. Path to an .xml file with a trained Head Pose Estimation model.";
static const char emotions_model_message[] = "Optional. Path to an .xml file with a trained Emotions Recognition model.";
static const char facial_landmarks_model_message[] = "Optional. Path to an .xml file with a trained Facial Landmarks Estimation model.";
static const char target_device_message[] = "Optional. Target device for Face Detection network (the list of available devices is shown below). "
                                            "Default value is CPU. The demo will look for a suitable plugin for a specified device.";
static const char target_device_message_ag[] = "Optional. Target device for Age/Gender Recognition network (the list of available devices is shown below). "
                                               "Default value is CPU. The demo will look for a suitable plugin for a specified device.";
static const char target_device_message_hp[] = "Optional. Target device for Head Pose Estimation network (the list of available devices is shown below). "
                                               "Default value is CPU. The demo will look for a suitable plugin for a specified device.";
static const char target_device_message_em[] = "Optional. Target device for Emotions Recognition network (the list of available devices is shown below). "
                                               "Default value is CPU. The demo will look for a suitable plugin for a specified device.";
static const char target_device_message_lm[] = "Optional. Target device for Facial Landmarks Estimation network "
                                               "(the list of available devices is shown below). Default value is CPU. The demo will look for a suitable plugin for device specified.";
static const char thresh_output_message[] = "Optional. Probability threshold for detections";
static const char bb_enlarge_coef_output_message[] = "Optional. Coefficient to enlarge/reduce the size of the bounding box around the detected face";
static const char raw_output_message[] = "Optional. Output inference results as raw values";
static const char no_show_message[] = "Optional. Don't show output.";
static const char dx_coef_output_message[] = "Optional. Coefficient to shift the bounding box around the detected face along the Ox axis";
static const char dy_coef_output_message[] = "Optional. Coefficient to shift the bounding box around the detected face along the Oy axis";
// TODO: Make this option valid for single image case
static const char loop_output_message[] = "Optional. Enable playing video on a loop";
static const char no_smooth_output_message[] = "Optional. Do not smooth person attributes";
static const char no_show_emotion_bar_message[] = "Optional. Do not show emotion bar";
static const char utilization_monitors_message[] = "Optional. List of monitors to show initially.";

// TODO: Support options:
// Set number of maximum simultaneously processed faces for Age/Gender,
// Head Pose Estimation, Emotions Recognition, Facial Landmarks Estimation networks

// Enabling dynamic batch size for Age/Gender, Head Pose Estimation, Emotions Recognition,
// Facial Landmarks Estimation networks

// Enabling per-layer performance report

// Enabling using of CPU custom layers.
// Enabling using of GPU custom kernels.

// Set maximum FPS for playing video

DEFINE_bool(h, false, help_message);
DEFINE_string(i, "", input_video_message);
DEFINE_string(m, "", face_detection_model_message);
DEFINE_string(m_ag, "", age_gender_model_message);
DEFINE_string(m_hp, "", head_pose_model_message);
DEFINE_string(m_em, "", emotions_model_message);
DEFINE_string(m_lm, "", facial_landmarks_model_message);
DEFINE_string(d, "CPU", target_device_message);
DEFINE_string(d_ag, "CPU", target_device_message_ag);
DEFINE_string(d_hp, "CPU", target_device_message_hp);
DEFINE_string(d_em, "CPU", target_device_message_em);
DEFINE_string(d_lm, "CPU", target_device_message_lm);
DEFINE_bool(r, false, raw_output_message);
DEFINE_double(t, 0.5, thresh_output_message);
DEFINE_double(bb_enlarge_coef, 1.2, bb_enlarge_coef_output_message);
DEFINE_bool(no_show, false, no_show_message);
DEFINE_double(dx_coef, 1, dx_coef_output_message);
DEFINE_double(dy_coef, 1, dy_coef_output_message);
DEFINE_bool(loop, false, loop_output_message);
DEFINE_bool(no_smooth, false, no_smooth_output_message);
DEFINE_bool(no_show_emotion_bar, false, no_show_emotion_bar_message);
DEFINE_string(u, "", utilization_monitors_message);

/**
* \brief This function shows a help message
*/

static void showUsage() {
    std::cout << std::endl;
    std::cout << "interactive_face_detection_demo_gapi [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                         " << help_message << std::endl;
    std::cout << "    -i \"<path>\"                " << input_video_message << std::endl;
    std::cout << "    -o \"<path>\"                " << output_message << std::endl;
    std::cout << "    -limit \"<num>\"             " << limit_message << std::endl;
    std::cout << "    -m \"<path>\"                " << face_detection_model_message<< std::endl;
    std::cout << "    -m_ag \"<path>\"             " << age_gender_model_message << std::endl;
    std::cout << "    -m_hp \"<path>\"             " << head_pose_model_message << std::endl;
    std::cout << "    -m_em \"<path>\"             " << emotions_model_message << std::endl;
    std::cout << "    -m_lm \"<path>\"             " << facial_landmarks_model_message << std::endl;
    std::cout << "    -d \"<device>\"              " << target_device_message << std::endl;
    std::cout << "    -d_ag \"<device>\"           " << target_device_message_ag << std::endl;
    std::cout << "    -d_hp \"<device>\"           " << target_device_message_hp << std::endl;
    std::cout << "    -d_em \"<device>\"           " << target_device_message_em << std::endl;
    std::cout << "    -d_lm \"<device>\"           " << target_device_message_lm << std::endl;
    std::cout << "    -no_show                   " << no_show_message << std::endl;
    std::cout << "    -r                         " << raw_output_message << std::endl;
    std::cout << "    -t                         " << thresh_output_message << std::endl;
    std::cout << "    -bb_enlarge_coef           " << bb_enlarge_coef_output_message << std::endl;
    std::cout << "    -dx_coef                   " << dx_coef_output_message << std::endl;
    std::cout << "    -dy_coef                   " << dy_coef_output_message << std::endl;
    std::cout << "    -loop                " << loop_output_message << std::endl;
    std::cout << "    -no_smooth                 " << no_smooth_output_message << std::endl;
    std::cout << "    -no_show_emotion_bar       " << no_show_emotion_bar_message << std::endl;
    std::cout << "    -u                         " << utilization_monitors_message << std::endl;
}
