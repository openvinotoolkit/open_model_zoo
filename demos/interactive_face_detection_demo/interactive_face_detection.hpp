// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>
#include <iostream>

/// @brief Message for help argument
static const char help_message[] = "Print a usage message";

/// @brief Message for images argument
static const char input_video_message[] = "Required. Path to a video file (specify \"cam\" to work with camera).";

/// @brief Message for images argument
static const char output_video_message[] = "Optional. Path to an output video file.";

/// @brief message for model argument
static const char face_detection_model_message[] = "Required. Path to an .xml file with a trained Face Detection model.";
static const char age_gender_model_message[] = "Optional. Path to an .xml file with a trained Age/Gender Recognition model.";
static const char head_pose_model_message[] = "Optional. Path to an .xml file with a trained Head Pose Estimation model.";
static const char emotions_model_message[] = "Optional. Path to an .xml file with a trained Emotions Recognition model.";
static const char facial_landmarks_model_message[] = "Optional. Path to an .xml file with a trained Facial Landmarks Estimation model.";

/// @brief Message for plugin argument
static const char plugin_message[] = "Plugin name. For example, CPU. If this parameter is specified, " \
"the demo will look for this plugin only.";

/// @brief Message for assigning face detection calculation to device
static const char target_device_message[] = "Optional. Target device for Face Detection network (the list of available devices is shown below). " \
"Default value is CPU. Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. " \
"The demo will look for a suitable plugin for a specified device.";

/// @brief Message for assigning age/gender calculation to device
static const char target_device_message_ag[] = "Optional. Target device for Age/Gender Recognition network (the list of available devices is shown below). " \
"Default value is CPU. Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. " \
"The demo will look for a suitable plugin for a specified device.";

/// @brief Message for assigning head pose calculation to device
static const char target_device_message_hp[] = "Optional. Target device for Head Pose Estimation network (the list of available devices is shown below). " \
"Default value is CPU. Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. " \
"The demo will look for a suitable plugin for a specified device.";

/// @brief Message for assigning emotions calculation to device
static const char target_device_message_em[] = "Optional. Target device for Emotions Recognition network (the list of available devices is shown below). " \
"Default value is CPU. Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. " \
"The demo will look for a suitable plugin for a specified device.";

/// @brief Message for assigning Facial Landmarks Estimation network to device
static const char target_device_message_lm[] = "Optional. Target device for Facial Landmarks Estimation network " \
"(the list of available devices is shown below). Default value is CPU. Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. " \
"The demo will look for a suitable plugin for device specified.";

/// @brief Message for the maximum number of simultaneously processed faces for Age Gender network
static const char num_batch_ag_message[] = "Optional. Number of maximum simultaneously processed faces for Age/Gender Recognition network " \
"(by default, it is 16)";

/// @brief Message for the maximum number of simultaneously processed faces for Head Pose network
static const char num_batch_hp_message[] = "Optional. Number of maximum simultaneously processed faces for Head Pose Estimation network " \
"(by default, it is 16)";

/// @brief Message for the maximum number of simultaneously processed faces for Emotions network
static const char num_batch_em_message[] = "Optional. Number of maximum simultaneously processed faces for Emotions Recognition network " \
"(by default, it is 16)";

/// @brief Message for the maximum number of simultaneously processed faces for Facial Landmarks Estimation network
static const char num_batch_lm_message[] = "Optional. Number of maximum simultaneously processed faces for Facial Landmarks Estimation network " \
"(by default, it is 16)";

/// @brief Message for dynamic batching support for AgeGender net
static const char dyn_batch_ag_message[] = "Optional. Enable dynamic batch size for Age/Gender Recognition network";

/// @brief Message for dynamic batching support for HeadPose net
static const char dyn_batch_hp_message[] = "Optional. Enable dynamic batch size for Head Pose Estimation network";

/// @brief Message for dynamic batching support for Emotions net
static const char dyn_batch_em_message[] = "Optional. Enable dynamic batch size for Emotions Recognition network";

/// @brief Message for dynamic batching support for Facial Landmarks Estimation network
static const char dyn_batch_lm_message[] = "Optional. Enable dynamic batch size for Facial Landmarks Estimation network";

/// @brief Message for performance counters
static const char performance_counter_message[] = "Optional. Enable per-layer performance report";

/// @brief Message for GPU custom kernels description
static const char custom_cldnn_message[] = "Required for GPU custom kernels. "\
"Absolute path to an .xml file with the kernels description.";

/// @brief Message for user library argument
static const char custom_cpu_library_message[] = "Required for CPU custom layers. " \
"Absolute path to a shared library with the kernels implementation.";

/// @brief Message for probability threshold argument
static const char thresh_output_message[] = "Optional. Probability threshold for detections";

/// @brief Message for face enlarge coefficient argument
static const char bb_enlarge_coef_output_message[] = "Optional. Coefficient to enlarge/reduce the size of the bounding box around the detected face";

/// @brief Message raw output flag
static const char raw_output_message[] = "Optional. Output inference results as raw values";

/// @brief Message do not wait for keypress after input stream completed
static const char no_wait_for_keypress_message[] = "Optional. Do not wait for key press in the end.";

/// @brief Message do not show processed video
static const char no_show_processed_video[] = "Optional. Do not show processed video.";

/// @brief Message for asynchronous mode
static const char async_message[] = "Optional. Enable asynchronous mode";

/// @brief Message for shifting coefficient by dx for detected faces
static const char dx_coef_output_message[] = "Optional. Coefficient to shift the bounding box around the detected face along the Ox axis";

/// @brief Message for shifting coefficient by dy for detected faces
static const char dy_coef_output_message[] = "Optional. Coefficient to shift the bounding box around the detected face along the Oy axis";

/// @brief Message for fps argument
static const char fps_output_message[] = "Optional. Maximum FPS for playing video";

/// @brief Message for looping video argument
static const char loop_video_output_message[] = "Optional. Enable playing video on a loop";

/// @brief Message for smooth argument
static const char no_smooth_output_message[] = "Optional. Do not smooth person attributes";

/// @brief Message for smooth argument
static const char no_show_emotion_bar_message[] = "Optional. Do not show emotion bar";

/// \brief Define flag for showing help message<br>
DEFINE_bool(h, false, help_message);

/// \brief Define parameter for set image file<br>
/// It is a required parameter
DEFINE_string(i, "", input_video_message);

/// \brief Define parameter for an output video file<br>
/// It is an optional parameter
DEFINE_string(o, "", output_video_message);

/// \brief Define parameter for Face Detection model file<br>
/// It is a required parameter
DEFINE_string(m, "", face_detection_model_message);

/// \brief Define parameter for Age Gender Recognition model file<br>
/// It is a optional parameter
DEFINE_string(m_ag, "", age_gender_model_message);

/// \brief Define parameter for Head Pose Estimation model file<br>
/// It is a optional parameter
DEFINE_string(m_hp, "", head_pose_model_message);

/// \brief Define parameter for Emotions Recognition model file<br>
/// It is a optional parameter
DEFINE_string(m_em, "", emotions_model_message);

/// \brief Define parameter for Facial Landmarks Estimation model file<br>
/// It is an optional parameter
DEFINE_string(m_lm, "", facial_landmarks_model_message);

/// \brief target device for Face Detection network<br>
DEFINE_string(d, "CPU", target_device_message);

/// \brief Define parameter for target device for Age/Gender Recognition network<br>
DEFINE_string(d_ag, "CPU", target_device_message_ag);

/// \brief Define parameter for target device for Head Pose Estimation network<br>
DEFINE_string(d_hp, "CPU", target_device_message_hp);

/// \brief Define parameter for target device for Emotions Recognition network<br>
DEFINE_string(d_em, "CPU", target_device_message_em);

/// \brief Define parameter for target device for Facial Landmarks Estimation network<br>
DEFINE_string(d_lm, "CPU", target_device_message_lm);

/// \brief Define parameter for maximum batch size for Age/Gender Recognition network<br>
DEFINE_uint32(n_ag, 16, num_batch_ag_message);

/// \brief Define parameter to enable dynamic batch size for Age/Gender Recognition network<br>
DEFINE_bool(dyn_ag, false, dyn_batch_ag_message);

/// \brief Define parameter for maximum batch size for Head Pose Estimation network<br>
DEFINE_uint32(n_hp, 16, num_batch_hp_message);

/// \brief Define parameter to enable dynamic batch size for Head Pose Estimation network<br>
DEFINE_bool(dyn_hp, false, dyn_batch_hp_message);

/// \brief Define parameter for maximum batch size for Emotions Recognition network<br>
DEFINE_uint32(n_em, 16, num_batch_em_message);

/// \brief Define parameter to enable dynamic batch size for Emotions Recognition network<br>
DEFINE_bool(dyn_em, false, dyn_batch_em_message);

/// \brief Define parameter for maximum batch size for Facial Landmarks Estimation network<br>
DEFINE_uint32(n_lm, 16, num_batch_em_message);

/// \brief Define parameter to enable dynamic batch size for Facial Landmarks Estimation network<br>
DEFINE_bool(dyn_lm, false, dyn_batch_em_message);

/// \brief Define parameter to enable per-layer performance report<br>
DEFINE_bool(pc, false, performance_counter_message);

/// @brief Define parameter for GPU custom kernels path<br>
/// Default is ./lib
DEFINE_string(c, "", custom_cldnn_message);

/// @brief Define parameter for absolute path to CPU library with user layers<br>
/// It is an optional parameter
DEFINE_string(l, "", custom_cpu_library_message);

/// \brief Define a flag to output raw scoring results<br>
/// It is an optional parameter
DEFINE_bool(r, false, raw_output_message);

/// \brief Define a parameter for probability threshold for detections<br>
/// It is an optional parameter
DEFINE_double(t, 0.5, thresh_output_message);

/// \brief Define a parameter to enlarge the bounding box around the detected face for more robust operation of face analytics networks<br>
/// It is an optional parameter
DEFINE_double(bb_enlarge_coef, 1.2, bb_enlarge_coef_output_message);

/// \brief Define a flag to disable keypress exit<br>
/// It is an optional parameter
DEFINE_bool(no_wait, false, no_wait_for_keypress_message);

/// \brief Define a flag to disable showing processed video<br>
/// It is an optional parameter
DEFINE_bool(no_show, false, no_show_processed_video);

/// \brief Define a flag to enable aynchronous execution<br>
/// It is an optional parameter
DEFINE_bool(async, false, async_message);

/// \brief Define a parameter to shift face bounding box by Ox for more robust operation of face analytics networks<br>
/// It is an optional parameter
DEFINE_double(dx_coef, 1, dx_coef_output_message);

/// \brief Define a parameter to shift face bounding box by Oy for more robust operation of face analytics networks<br>
/// It is an optional parameter
DEFINE_double(dy_coef, 1, dy_coef_output_message);

/// \brief Define a parameter to play video with defined fps<br>
/// It is an optional parameter
DEFINE_double(fps, -1, fps_output_message);

/// \brief Define a flag to loop video<br>
/// It is an optional parameter
DEFINE_bool(loop_video, false, loop_video_output_message);

/// \brief Define a flag to disable smoothing person attributes<br>
/// It is an optional parameter
DEFINE_bool(no_smooth, false, no_smooth_output_message);

/// \brief Define a flag to disable showing emotion bar<br>
/// It is an optional parameter
DEFINE_bool(no_show_emotion_bar, false, no_show_emotion_bar_message);


/**
* \brief This function shows a help message
*/

static void showUsage() {
    std::cout << std::endl;
    std::cout << "interactive_face_detection [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                         " << help_message << std::endl;
    std::cout << "    -i \"<path>\"                " << input_video_message << std::endl;
    std::cout << "    -o \"<path>\"                " << output_video_message << std::endl;
    std::cout << "    -m \"<path>\"                " << face_detection_model_message<< std::endl;
    std::cout << "    -m_ag \"<path>\"             " << age_gender_model_message << std::endl;
    std::cout << "    -m_hp \"<path>\"             " << head_pose_model_message << std::endl;
    std::cout << "    -m_em \"<path>\"             " << emotions_model_message << std::endl;
    std::cout << "    -m_lm \"<path>\"             " << facial_landmarks_model_message << std::endl;
    std::cout << "      -l \"<absolute_path>\"     " << custom_cpu_library_message << std::endl;
    std::cout << "          Or" << std::endl;
    std::cout << "      -c \"<absolute_path>\"     " << custom_cldnn_message << std::endl;
    std::cout << "    -d \"<device>\"              " << target_device_message << std::endl;
    std::cout << "    -d_ag \"<device>\"           " << target_device_message_ag << std::endl;
    std::cout << "    -d_hp \"<device>\"           " << target_device_message_hp << std::endl;
    std::cout << "    -d_em \"<device>\"           " << target_device_message_em << std::endl;
    std::cout << "    -d_lm \"<device>\"           " << target_device_message_lm << std::endl;
    std::cout << "    -n_ag \"<num>\"              " << num_batch_ag_message << std::endl;
    std::cout << "    -n_hp \"<num>\"              " << num_batch_hp_message << std::endl;
    std::cout << "    -n_em \"<num>\"              " << num_batch_em_message << std::endl;
    std::cout << "    -n_lm \"<num>\"              " << num_batch_lm_message << std::endl;
    std::cout << "    -dyn_ag                    " << dyn_batch_ag_message << std::endl;
    std::cout << "    -dyn_hp                    " << dyn_batch_hp_message << std::endl;
    std::cout << "    -dyn_em                    " << dyn_batch_em_message << std::endl;
    std::cout << "    -dyn_lm                    " << dyn_batch_lm_message << std::endl;
    std::cout << "    -async                     " << async_message << std::endl;
    std::cout << "    -no_wait                   " << no_wait_for_keypress_message << std::endl;
    std::cout << "    -no_show                   " << no_show_processed_video << std::endl;
    std::cout << "    -pc                        " << performance_counter_message << std::endl;
    std::cout << "    -r                         " << raw_output_message << std::endl;
    std::cout << "    -t                         " << thresh_output_message << std::endl;
    std::cout << "    -bb_enlarge_coef           " << bb_enlarge_coef_output_message << std::endl;
    std::cout << "    -dx_coef                   " << dx_coef_output_message << std::endl;
    std::cout << "    -dy_coef                   " << dy_coef_output_message << std::endl;
    std::cout << "    -fps                       " << fps_output_message << std::endl;
    std::cout << "    -loop_video                " << loop_video_output_message << std::endl;
    std::cout << "    -no_smooth                 " << no_smooth_output_message << std::endl;
    std::cout << "    -no_show_emotion_bar       " << no_show_emotion_bar_message << std::endl;
}
