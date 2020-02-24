// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>

static const char help_message[] = "Print a usage message.";
static const char video_message[] = "Required. Path to a video or image file. Default value is \"cam\" to work with camera.";
static const char person_action_detection_model_message[] = "Required. Path to the Person/Action Detection Retail model (.xml) file.";
static const char face_detection_model_message[] = "Required. Path to the Face Detection Retail model (.xml) file.";
static const char facial_landmarks_model_message[] = "Required. Path to the Facial Landmarks Regression Retail model (.xml) file.";
static const char face_reid_model_message[] = "Required. Path to the Face Reidentification Retail model (.xml) file.";
static const char target_device_message_action_detection[] = "Optional. Specify the target device for Person/Action Detection Retail "
                                                             "(the list of available devices is shown below).Default value is CPU. "
                                                             "Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. "
                                                             "The application looks for a suitable plugin for the specified device.";
static const char target_device_message_face_detection[] = "Optional. Specify the target device for Face Detection Retail "
                                                           "(the list of available devices is shown below).Default value is CPU. "
                                                           "Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. "
                                                           "The application looks for a suitable plugin for the specified device.";
static const char target_device_message_landmarks_regression[] = "Optional. Specify the target device for Landmarks Regression Retail "
                                                                 "(the list of available devices is shown below).Default value is CPU. "
                                                                 "Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. "
                                                                 "The application looks for a suitable plugin for the specified device.";
static const char target_device_message_face_reid[] = "Optional. Specify the target device for Face Reidentification Retail "
                                                      "(the list of available devices is shown below).Default value is CPU. "
                                                      "Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. "
                                                      "The application looks for a suitable plugin for the specified device.";
static const char greedy_reid_matching_message[] = "Optional. Use faster greedy matching algorithm in face reid.";
static const char performance_counter_message[] = "Optional. Enables per-layer performance statistics.";
static const char custom_cldnn_message[] = "Optional. For GPU custom kernels, if any. "
                                           "Absolute path to an .xml file with the kernels description.";
static const char custom_cpu_library_message[] = "Optional. For CPU custom layers, if any. "
                                                 "Absolute path to a shared library with the kernels implementation.";
static const char face_threshold_output_message[] = "Optional. Probability threshold for face detections.";
static const char person_threshold_output_message[] = "Optional. Probability threshold for person/action detection.";
static const char action_threshold_output_message[] = "Optional. Probability threshold for action recognition.";
static const char threshold_output_message_face_reid[] = "Optional. Cosine distance threshold between two vectors for face reidentification.";
static const char reid_gallery_path_message[] = "Optional. Path to a faces gallery in .json format.";
static const char output_video_message[] = "Optional. File to write output video with visualization to.";
static const char act_stat_output_message[] = "Optional. Output file name to save per-person action statistics in.";
static const char raw_output_message[] = "Optional. Output Inference results as raw values.";
static const char no_show_processed_video[] = "Optional. Do not show processed video.";
static const char input_image_height_output_message[] = "Optional. Input image height for face detector.";
static const char input_image_width_output_message[] = "Optional. Input image width for face detector.";
static const char expand_ratio_output_message[] = "Optional. Expand ratio for bbox before face recognition.";
static const char last_frame_message[] = "Optional. Last frame number to handle in demo. If negative, handle all input video.";
static const char teacher_id_message[] = "Optional. ID of a teacher. You must also set a faces gallery parameter (-fg) to use it.";
static const char min_action_duration_message[] = "Optional. Minimum action duration in seconds.";
static const char same_action_time_delta_message[] = "Optional. Maximum time difference between actions in seconds.";
static const char student_actions_message[] = "Optional. List of student actions separated by a comma.";
static const char top_actions_message[] = "Optional. List of student actions (for top-k mode) separated by a comma.";
static const char teacher_actions_message[] = "Optional. List of teacher actions separated by a comma.";
static const char target_action_name_message[] = "Optional. Target action name.";
static const char target_actions_num_message[] = "Optional. Number of first K students. If this parameter is positive,"
                                                 "the demo detects first K persons with the action, pointed by the parameter 'top_id'";
static const char crop_gallery_message[] = "Optional. Crop images during faces gallery creation.";
static const char face_threshold_registration_output_message[] = "Optional. Probability threshold for face detections during database registration.";
static const char min_size_fr_reg_output_message[] = "Optional. Minimum input size for faces during database registration.";
static const char act_det_output_message[] = "Optional. Output file name to save per-person action detections in.";
static const char tracker_smooth_size_message[] = "Optional. Number of frames to smooth actions.";
static const char utilization_monitors_message[] = "Optional. List of monitors to show initially.";

DEFINE_bool(h, false, help_message);
DEFINE_string(i, "cam", video_message);
DEFINE_string(m_act, "", person_action_detection_model_message);
DEFINE_string(m_fd, "", face_detection_model_message);
DEFINE_string(m_lm, "", facial_landmarks_model_message);
DEFINE_string(m_reid, "", face_reid_model_message);
DEFINE_string(d_act, "CPU", target_device_message_action_detection);
DEFINE_string(d_fd, "CPU", target_device_message_face_detection);
DEFINE_string(d_lm, "CPU", target_device_message_landmarks_regression);
DEFINE_string(d_reid, "CPU", target_device_message_face_reid);
DEFINE_bool(greedy_reid_matching, false, greedy_reid_matching_message);
DEFINE_bool(pc, false, performance_counter_message);
DEFINE_string(c, "", custom_cldnn_message);
DEFINE_string(l, "", custom_cpu_library_message);
DEFINE_string(ad, "", act_stat_output_message);
DEFINE_bool(r, false, raw_output_message);
DEFINE_double(t_ad, 0.3, person_threshold_output_message);
DEFINE_double(t_ar, 0.75, action_threshold_output_message);
DEFINE_double(t_fd, 0.6, face_threshold_output_message);
DEFINE_double(t_reid, 0.7, threshold_output_message_face_reid);
DEFINE_string(fg, "", reid_gallery_path_message);
DEFINE_string(out_v, "", output_video_message);
DEFINE_bool(no_show, false, no_show_processed_video);
DEFINE_int32(inh_fd, 600, input_image_height_output_message);
DEFINE_int32(inw_fd, 600, input_image_width_output_message);
DEFINE_double(exp_r_fd, 1.15, face_threshold_output_message);
DEFINE_int32(last_frame, -1, last_frame_message);
DEFINE_string(teacher_id, "", teacher_id_message);
DEFINE_double(min_ad, 1.0, min_action_duration_message);
DEFINE_double(d_ad, 1.0, same_action_time_delta_message);
DEFINE_string(student_ac, "sitting,standing,raising_hand", student_actions_message);
DEFINE_string(top_ac, "sitting,raising_hand", top_actions_message);
DEFINE_string(teacher_ac, "standing,writing,demonstrating", teacher_actions_message);
DEFINE_string(top_id, "raising_hand", target_action_name_message);
DEFINE_int32(a_top, -1, target_actions_num_message);
DEFINE_bool(crop_gallery, false, crop_gallery_message);
DEFINE_double(t_reg_fd, 0.9, face_threshold_registration_output_message);
DEFINE_int32(min_size_fr, 128, min_size_fr_reg_output_message);
DEFINE_string(al, "", act_det_output_message);
DEFINE_int32(ss_t, -1, tracker_smooth_size_message);
DEFINE_string(u, "", utilization_monitors_message);

/**
* @brief This function show a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "smart_classroom_demo [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                             " << help_message << std::endl;
    std::cout << "    -i '<path>'                    " << video_message << std::endl;
    std::cout << "    -m_act '<path>'                " << person_action_detection_model_message << std::endl;
    std::cout << "    -m_fd '<path>'                 " << face_detection_model_message << std::endl;
    std::cout << "    -m_lm '<path>'                 " << facial_landmarks_model_message << std::endl;
    std::cout << "    -m_reid '<path>'               " << face_reid_model_message << std::endl;
    std::cout << "    -l '<absolute_path>'           " << custom_cpu_library_message << std::endl;
    std::cout << "          Or" << std::endl;
    std::cout << "    -c '<absolute_path>'           " << custom_cldnn_message << std::endl;
    std::cout << "    -d_act '<device>'              " << target_device_message_action_detection << std::endl;
    std::cout << "    -d_fd '<device>'               " << target_device_message_face_detection << std::endl;
    std::cout << "    -d_lm '<device>'               " << target_device_message_landmarks_regression << std::endl;
    std::cout << "    -d_reid '<device>'             " << target_device_message_face_reid << std::endl;
    std::cout << "    -out_v  '<path>'               " << output_video_message << std::endl;
    std::cout << "    -greedy_reid_matching          " << greedy_reid_matching_message << std::endl;
    std::cout << "    -pc                            " << performance_counter_message << std::endl;
    std::cout << "    -r                             " << raw_output_message << std::endl;
    std::cout << "    -ad                            " << act_stat_output_message << std::endl;
    std::cout << "    -t_ad                          " << person_threshold_output_message << std::endl;
    std::cout << "    -t_ar                          " << action_threshold_output_message << std::endl;
    std::cout << "    -t_fd                          " << face_threshold_output_message << std::endl;
    std::cout << "    -inh_fd                        " << input_image_height_output_message << std::endl;
    std::cout << "    -inw_fd                        " << input_image_width_output_message << std::endl;
    std::cout << "    -exp_r_fd                      " << expand_ratio_output_message << std::endl;
    std::cout << "    -t_reid                        " << threshold_output_message_face_reid << std::endl;
    std::cout << "    -fg                            " << reid_gallery_path_message << std::endl;
    std::cout << "    -teacher_id                    " << teacher_id_message << std::endl;
    std::cout << "    -no_show                       " << no_show_processed_video << std::endl;
    std::cout << "    -last_frame                    " << last_frame_message << std::endl;
    std::cout << "    -min_ad                        " << min_action_duration_message << std::endl;
    std::cout << "    -d_ad                          " << same_action_time_delta_message << std::endl;
    std::cout << "    -student_ac                    " << student_actions_message << std::endl;
    std::cout << "    -top_ac                        " << top_actions_message << std::endl;
    std::cout << "    -teacher_ac                    " << teacher_actions_message << std::endl;
    std::cout << "    -a_id                          " << target_action_name_message << std::endl;
    std::cout << "    -a_top                         " << target_actions_num_message << std::endl;
    std::cout << "    -crop_gallery                  " << crop_gallery_message << std::endl;
    std::cout << "    -t_reg_fd                      " << face_threshold_registration_output_message << std::endl;
    std::cout << "    -min_size_fr                   " << min_size_fr_reg_output_message << std::endl;
    std::cout << "    -al                            " << act_det_output_message << std::endl;
    std::cout << "    -ss_t                          " << tracker_smooth_size_message << std::endl;
    std::cout << "    -u                             " << utilization_monitors_message << std::endl;
}
