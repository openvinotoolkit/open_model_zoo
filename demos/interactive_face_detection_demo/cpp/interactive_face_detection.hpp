// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>
#include <iostream>

#include <utils/default_flags.hpp>

DEFINE_INPUT_FLAGS
DEFINE_OUTPUT_FLAGS

static const char help_message[] = "Print a usage message";
static const char face_detection_model_message[] = "Required. Path to an .xml file with a trained Face Detection model.";
static const char age_gender_model_message[] = "Optional. Path to an .xml file with a trained Age/Gender Recognition model.";
static const char head_pose_model_message[] = "Optional. Path to an .xml file with a trained Head Pose Estimation model.";
static const char emotions_model_message[] = "Optional. Path to an .xml file with a trained Emotions Recognition model.";
static const char facial_landmarks_model_message[] = "Optional. Path to an .xml file with a trained Facial Landmarks Estimation model.";
static const char antispoofing_model_message[] = "Optional. Path to an .xml file with a trained Antispoofing Classification model.";
static const char target_device_message[] = "Optional. Target device for Face Detection network (the list of available devices is shown below). "
                                            "Default value is CPU. Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. "
                                            "The demo will look for a suitable plugin for a specified device.";
static const char target_device_message_ag[] = "Optional. Target device for Age/Gender Recognition network (the list of available devices is shown below). "
                                               "Default value is CPU. Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. "
                                               "The demo will look for a suitable plugin for a specified device.";
static const char target_device_message_hp[] = "Optional. Target device for Head Pose Estimation network (the list of available devices is shown below). "
                                               "Default value is CPU. Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. "
                                               "The demo will look for a suitable plugin for a specified device.";
static const char target_device_message_em[] = "Optional. Target device for Emotions Recognition network (the list of available devices is shown below). "
                                               "Default value is CPU. Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. "
                                               "The demo will look for a suitable plugin for a specified device.";
static const char target_device_message_lm[] = "Optional. Target device for Facial Landmarks Estimation network "
                                               "(the list of available devices is shown below). Default value is CPU. Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. "
                                               "The demo will look for a suitable plugin for a specified device.";
static const char target_device_message_am[] = "Optional. Target device for Antispoofing Classification network (the list of available devices is shown below). "
                                               "Default value is CPU. Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. "
                                               "The demo will look for a suitable plugin for a specified device.";
static const char num_batch_ag_message[] = "Optional. Number of maximum simultaneously processed faces for Age/Gender Recognition network "
                                           "(by default, it is 16)";
static const char num_batch_hp_message[] = "Optional. Number of maximum simultaneously processed faces for Head Pose Estimation network "
                                           "(by default, it is 16)";
static const char num_batch_em_message[] = "Optional. Number of maximum simultaneously processed faces for Emotions Recognition network "
                                           "(by default, it is 16)";
static const char num_batch_lm_message[] = "Optional. Number of maximum simultaneously processed faces for Facial Landmarks Estimation network "
                                           "(by default, it is 16)";
static const char num_batch_am_message[] = "Optional. Number of maximum simultaneously processed faces for Antispoofing Classification network "
                                           "(by default, it is 16)";
static const char dyn_batch_ag_message[] = "Optional. Enable dynamic batch size for Age/Gender Recognition network";
static const char dyn_batch_hp_message[] = "Optional. Enable dynamic batch size for Head Pose Estimation network";
static const char dyn_batch_em_message[] = "Optional. Enable dynamic batch size for Emotions Recognition network";
static const char dyn_batch_lm_message[] = "Optional. Enable dynamic batch size for Facial Landmarks Estimation network";
static const char dyn_batch_am_message[] = "Optional. Enable dynamic batch size for Antispoofing Classification network";
static const char performance_counter_message[] = "Optional. Enable per-layer performance report";
static const char custom_cldnn_message[] = "Required for GPU custom kernels. "
                                           "Absolute path to an .xml file with the kernels description.";
static const char custom_cpu_library_message[] = "Required for CPU custom layers. "
                                                 "Absolute path to a shared library with the kernels implementation.";
static const char thresh_output_message[] = "Optional. Probability threshold for detections";
static const char bb_enlarge_coef_output_message[] = "Optional. Coefficient to enlarge/reduce the size of the bounding box around the detected face";
static const char raw_output_message[] = "Optional. Output inference results as raw values";
static const char no_show_message[] = "Optional. Don't show output.";
static const char async_message[] = "Optional. Enable asynchronous mode";
static const char dx_coef_output_message[] = "Optional. Coefficient to shift the bounding box around the detected face along the Ox axis";
static const char dy_coef_output_message[] = "Optional. Coefficient to shift the bounding box around the detected face along the Oy axis";
static const char fps_output_message[] = "Optional. Maximum FPS for playing video";
static const char no_smooth_output_message[] = "Optional. Do not smooth person attributes";
static const char no_show_emotion_bar_message[] = "Optional. Do not show emotion bar";
static const char utilization_monitors_message[] = "Optional. List of monitors to show initially.";

DEFINE_bool(h, false, help_message);
DEFINE_string(m, "", face_detection_model_message);
DEFINE_string(m_ag, "", age_gender_model_message);
DEFINE_string(m_hp, "", head_pose_model_message);
DEFINE_string(m_em, "", emotions_model_message);
DEFINE_string(m_lm, "", facial_landmarks_model_message);
DEFINE_string(m_am, "", antispoofing_model_message);
DEFINE_string(d, "CPU", target_device_message);
DEFINE_string(d_ag, "CPU", target_device_message_ag);
DEFINE_string(d_hp, "CPU", target_device_message_hp);
DEFINE_string(d_em, "CPU", target_device_message_em);
DEFINE_string(d_lm, "CPU", target_device_message_lm);
DEFINE_string(d_am, "CPU", target_device_message_am);
DEFINE_uint32(n_ag, 16, num_batch_ag_message);
DEFINE_bool(dyn_ag, false, dyn_batch_ag_message);
DEFINE_uint32(n_hp, 16, num_batch_hp_message);
DEFINE_bool(dyn_hp, false, dyn_batch_hp_message);
DEFINE_uint32(n_em, 16, num_batch_em_message);
DEFINE_bool(dyn_em, false, dyn_batch_em_message);
DEFINE_uint32(n_lm, 16, num_batch_em_message);
DEFINE_bool(dyn_lm, false, dyn_batch_em_message);
DEFINE_uint32(n_am, 16, num_batch_am_message);
DEFINE_bool(dyn_am, false, dyn_batch_am_message);
DEFINE_bool(pc, false, performance_counter_message);
DEFINE_string(c, "", custom_cldnn_message);
DEFINE_string(l, "", custom_cpu_library_message);
DEFINE_bool(r, false, raw_output_message);
DEFINE_double(t, 0.5, thresh_output_message);
DEFINE_double(bb_enlarge_coef, 1.2, bb_enlarge_coef_output_message);
DEFINE_bool(no_show, false, no_show_message);
DEFINE_bool(async, false, async_message);
DEFINE_double(dx_coef, 1, dx_coef_output_message);
DEFINE_double(dy_coef, 1, dy_coef_output_message);
DEFINE_double(fps, -std::numeric_limits<double>::infinity(), fps_output_message);
DEFINE_bool(no_smooth, false, no_smooth_output_message);
DEFINE_bool(no_show_emotion_bar, false, no_show_emotion_bar_message);
DEFINE_string(u, "", utilization_monitors_message);


/**
* \brief This function shows a help message
*/

static void showUsage() {
    std::cout << std::endl;
    std::cout << "interactive_face_detection_demo [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                         " << help_message << std::endl;
    std::cout << "    -i                         " << input_message << std::endl;
    std::cout << "    -loop                      " << loop_message << std::endl;
    std::cout << "    -o \"<path>\"                " << output_message << std::endl;
    std::cout << "    -limit \"<num>\"             " << limit_message << std::endl;
    std::cout << "    -m \"<path>\"                " << face_detection_model_message << std::endl;
    std::cout << "    -m_ag \"<path>\"             " << age_gender_model_message << std::endl;
    std::cout << "    -m_hp \"<path>\"             " << head_pose_model_message << std::endl;
    std::cout << "    -m_em \"<path>\"             " << emotions_model_message << std::endl;
    std::cout << "    -m_lm \"<path>\"             " << facial_landmarks_model_message << std::endl;
    std::cout << "    -m_am \"<path>\"             " << antispoofing_model_message << std::endl;
    std::cout << "      -l \"<absolute_path>\"     " << custom_cpu_library_message << std::endl;
    std::cout << "          Or" << std::endl;
    std::cout << "      -c \"<absolute_path>\"     " << custom_cldnn_message << std::endl;
    std::cout << "    -d \"<device>\"              " << target_device_message << std::endl;
    std::cout << "    -d_ag \"<device>\"           " << target_device_message_ag << std::endl;
    std::cout << "    -d_hp \"<device>\"           " << target_device_message_hp << std::endl;
    std::cout << "    -d_em \"<device>\"           " << target_device_message_em << std::endl;
    std::cout << "    -d_lm \"<device>\"           " << target_device_message_lm << std::endl;
    std::cout << "    -d_am \"<device>\"           " << target_device_message_am << std::endl;
    std::cout << "    -n_ag \"<num>\"              " << num_batch_ag_message << std::endl;
    std::cout << "    -n_hp \"<num>\"              " << num_batch_hp_message << std::endl;
    std::cout << "    -n_em \"<num>\"              " << num_batch_em_message << std::endl;
    std::cout << "    -n_lm \"<num>\"              " << num_batch_lm_message << std::endl;
    std::cout << "    -n_am \"<num>\"              " << num_batch_am_message << std::endl;
    std::cout << "    -dyn_ag                    " << dyn_batch_ag_message << std::endl;
    std::cout << "    -dyn_hp                    " << dyn_batch_hp_message << std::endl;
    std::cout << "    -dyn_em                    " << dyn_batch_em_message << std::endl;
    std::cout << "    -dyn_lm                    " << dyn_batch_lm_message << std::endl;
    std::cout << "    -dyn_am                    " << dyn_batch_am_message << std::endl;
    std::cout << "    -async                     " << async_message << std::endl;
    std::cout << "    -no_show                   " << no_show_message << std::endl;
    std::cout << "    -pc                        " << performance_counter_message << std::endl;
    std::cout << "    -r                         " << raw_output_message << std::endl;
    std::cout << "    -t                         " << thresh_output_message << std::endl;
    std::cout << "    -bb_enlarge_coef           " << bb_enlarge_coef_output_message << std::endl;
    std::cout << "    -dx_coef                   " << dx_coef_output_message << std::endl;
    std::cout << "    -dy_coef                   " << dy_coef_output_message << std::endl;
    std::cout << "    -fps                       " << fps_output_message << std::endl;
    std::cout << "    -no_smooth                 " << no_smooth_output_message << std::endl;
    std::cout << "    -no_show_emotion_bar       " << no_show_emotion_bar_message << std::endl;
    std::cout << "    -u                         " << utilization_monitors_message << std::endl;
}
