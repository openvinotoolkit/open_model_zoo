// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>

/// @brief Message for help argument
static const char help_message[] = "Print a usage message.";

/// @brief Message for images argument
static const char video_message[] = "Required. Path to a video or image file. Default value is \"cam\" to work with camera.";

/// @brief Message for model argument
static const char person_action_detection_model_message[] = "Required. Path to the Person/Action Detection Retail model (.xml) file.";
static const char face_detection_model_message[] = "Required. Path to the Face Detection Retail model (.xml) file.";
static const char facial_landmarks_model_message[] = "Required. Path to the Facial Landmarks Regression Retail model (.xml) file.";
static const char face_reid_model_message[] = "Required. Path to the Face Reidentification Retail model (.xml) file.";

/// @brief Message for assigning Person/Action detection inference to device
static const char target_device_message_action_detection[] = "Optional. Specify the target device for Person/Action Detection Retail "\
                                                             "(the list of available devices is shown below).Default value is CPU. " \
                                                             "Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. " \
                                                             "The application looks for a suitable plugin for the specified device.";

/// @brief Message for assigning Face Detection inference to device
static const char target_device_message_face_detection[] = "Optional. Specify the target device for Face Detection Retail "\
                                                           "(the list of available devices is shown below).Default value is CPU. " \
                                                           "Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. " \
                                                           "The application looks for a suitable plugin for the specified device.";

/// @brief Message for assigning Landmarks Regression retail inference to device
static const char target_device_message_landmarks_regression[] = "Optional. Specify the target device for Landmarks Regression Retail "\
                                                                 "(the list of available devices is shown below).Default value is CPU. " \
                                                                 "Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. " \
                                                                 "The application looks for a suitable plugin for the specified device.";

/// @brief Message for assigning Face Reidentification retail inference to device
static const char target_device_message_face_reid[] = "Optional. Specify the target device for Face Reidentification Retail "\
                                                      "(the list of available devices is shown below).Default value is CPU. " \
                                                      "Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. " \
                                                      "The application looks for a suitable plugin for the specified device.";

/// @brief Message for performance counters
static const char performance_counter_message[] = "Optional. Enables per-layer performance statistics.";

/// @brief Message for clDNN custom kernels desc
static const char custom_cldnn_message[] = "Optional. For GPU custom kernels, if any. "\
"Absolute path to an .xml file with the kernels description.";

/// @brief Message for user library argument
static const char custom_cpu_library_message[] = "Optional. For CPU custom layers, if any. " \
"Absolute path to a shared library with the kernels implementation.";

/// @brief Message for probability threshold argument for face detections
static const char face_threshold_output_message[] = "Optional. Probability threshold for face detections.";

/// @brief Message for probability threshold argument for person/action detection
static const char person_threshold_output_message[] = "Optional. Probability threshold for person/action detection.";

// @brief Message for probability threshold argument for action recognition
static const char action_threshold_output_message[] = "Optional. Probability threshold for action recognition.";

/// @brief Message for cosine distance threshold for face reidentification
static const char threshold_output_message_face_reid[] = "Optional. Cosine distance threshold between two vectors for face reidentification.";

/// @brief Message for faces gallery path
static const char reid_gallery_path_message[] = "Optional. Path to a faces gallery in .json format.";

/// @brief Message for output video path
static const char output_video_message[] = "Optional. File to write output video with visualization to.";

/// @brief Message action statistics output flag
static const char act_stat_output_message[] = "Optional. Output file name to save per-person action statistics in.";

/// @brief Message raw output flag
static const char raw_output_message[] = "Optional. Output Inference results as raw values.";

/// @brief Message no show processed video
static const char no_show_processed_video[] = "Optional. Do not show processed video.";

/// @brief Message input image height for face detector
static const char input_image_height_output_message[] = "Optional. Input image height for face detector.";

/// @brief Message input image width for face detector
static const char input_image_width_output_message[] = "Optional. Input image width for face detector.";

/// @brief Message expand ratio for bbox
static const char expand_ratio_output_message[] = "Optional. Expand ratio for bbox before face recognition.";

/// @brief Message last frame number to handle
static const char last_frame_message[] = "Optional. Last frame number to handle in demo. If negative, handle all input video.";

/// @brief Message teacher id
static const char teacher_id_message[] = "Optional. ID of a teacher. You must also set a faces gallery parameter (-fg) to use it.";

/// @brief Message min action duration
static const char min_action_duration_message[] = "Optional. Minimum action duration in seconds.";

/// @brief Message same action time delta
static const char same_action_time_delta_message[] = "Optional. Maximum time difference between actions in seconds.";

/// @brief Message student actions
static const char student_actions_message[] = "Optional. List of student actions separated by a comma.";

/// @brief Message student actions for top-k mode
static const char top_actions_message[] = "Optional. List of student actions (for top-k mode) separated by a comma.";

/// @brief Message teacher actions
static const char teacher_actions_message[] = "Optional. List of teacher actions separated by a comma.";

/// @brief Message for target action name for top-k mode
static const char target_action_name_message[] = "Optional. Target action name.";

/// @brief Message for number of target actions
static const char target_actions_num_message[] = "Optional. Number of first K students. If this parameter is positive,"\
"the demo detects first K persons with the action, pointed by the parameter 'top_id'";

/// @brief Message crop gallery
static const char crop_gallery_message[] = "Optional. Crop images during faces gallery creation.";

/// @brief Message for probability threshold argument for face detections during database registration.
static const char face_threshold_registration_output_message[] = "Optional. Probability threshold for face detections during database registration.";

/// @brief Message for minumum input size for faces database registration.
static const char min_size_fr_reg_output_message[] = "Optional. Minimum input size for faces during database registration.";

/// @brief Message action statistics output flag
static const char act_det_output_message[] = "Optional. Output file name to save per-person action detections in.";

/// @brief Message for number of frames for action tracker
static const char tracker_smooth_size_message[] = "Optional. Number of frames to smooth actions.";

/// @brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// @brief Define parameter for set image file <br>
/// It is a required parameter
DEFINE_string(i, "cam", video_message);

/// @brief Define parameter for person/action detection model file <br>
/// It is a required parameter
DEFINE_string(m_act, "", person_action_detection_model_message);

/// @brief Define parameter for face detection model file <br>
/// It is a required parameter
DEFINE_string(m_fd, "", face_detection_model_message);

/// @brief Define parameter for facial landmarks model file <br>
/// It is a required parameter
DEFINE_string(m_lm, "", facial_landmarks_model_message);

/// @brief Define parameter for face reidentification model file <br>
/// It is a required parameter
DEFINE_string(m_reid, "", face_reid_model_message);

/// @brief device the target device for person/action detection infer on <br>
DEFINE_string(d_act, "CPU", target_device_message_action_detection);

/// @brief device the target device for face detection on <br>
DEFINE_string(d_fd, "CPU", target_device_message_face_detection);

/// @brief device the target device for facial landnmarks regression infer on <br>
DEFINE_string(d_lm, "CPU", target_device_message_landmarks_regression);

/// @brief device the target device for face reidentification infer on <br>
DEFINE_string(d_reid, "CPU", target_device_message_face_reid);

/// @brief Enable per-layer performance report
DEFINE_bool(pc, false, performance_counter_message);

/// @brief clDNN custom kernels path <br>
/// Default is ./lib
DEFINE_string(c, "", custom_cldnn_message);

/// @brief Absolute path to CPU library with user layers <br>
/// It is a optional parameter
DEFINE_string(l, "", custom_cpu_library_message);

/// @brief Output file name to save per-person action statistics in.
/// It is an optional parameter
DEFINE_string(ad, "", act_stat_output_message);

/// @brief Flag to output raw pipeline results<br>
/// It is an optional parameter
DEFINE_bool(r, false, raw_output_message);

/// @brief Define probability threshold for person/action detection <br>
/// It is an optional parameter
DEFINE_double(t_ad, 0.3, person_threshold_output_message);

/// @brief Define probability threshold for action recognition <br>
/// It is an optional parameter
DEFINE_double(t_ar, 0.75, action_threshold_output_message);

/// @brief Define probability threshold for face detections <br>
/// It is an optional parameter
DEFINE_double(t_fd, 0.6, face_threshold_output_message);

/// @brief Define cosine distance threshold for face reid <br>
/// It is an optional parameter
DEFINE_double(t_reid, 0.7, threshold_output_message_face_reid);

/// @brief Path to a faces gallery for reid <br>
/// It is a optional parameter
DEFINE_string(fg, "", reid_gallery_path_message);

/// @brief File to write output video with visualization to.
/// It is a optional parameter
DEFINE_string(out_v, "", output_video_message);

/// @brief Flag to disable processed video showing<br>
/// It is an optional parameter
DEFINE_bool(no_show, false, no_show_processed_video);

/// @brief Input image height for face detector<br>
/// It is an optional parameter
DEFINE_int32(inh_fd, 600, input_image_height_output_message);

/// @brief Input image width for face detector<br>
/// It is an optional parameter
DEFINE_int32(inw_fd, 600, input_image_width_output_message);

/// @brief Expand ratio for bbox before face recognition<br>
/// It is an optional parameter
DEFINE_double(exp_r_fd, 1.15, face_threshold_output_message);

/// @brief Input image height for face detector<br>
/// It is an optional parameter
DEFINE_int32(last_frame, -1, last_frame_message);

/// @brief Label of teacher<br>
/// It is an optional parameter
DEFINE_string(teacher_id, "", teacher_id_message);

/// @brief Min action duration<br>
/// It is an optional parameter
DEFINE_double(min_ad, 1.0, min_action_duration_message);

/// @brief Same action time delta<br>
/// It is an optional parameter
DEFINE_double(d_ad, 1.0, same_action_time_delta_message);

/// @brief Labels of student actions<br>
/// It is an optional parameter
DEFINE_string(student_ac, "sitting,standing,raising_hand", student_actions_message);

/// @brief Labels of student actions for top-k mode<br>
/// It is an optional parameter
DEFINE_string(top_ac, "sitting,raising_hand", top_actions_message);

/// @brief Labels of teacher actions<br>
/// It is an optional parameter
DEFINE_string(teacher_ac, "standing,writing,demonstrating", teacher_actions_message);

/// @brief Define target action name for top-k mode <br>
/// It is an optional parameter
DEFINE_string(top_id, "raising_hand", target_action_name_message);

/// @brief Define maximal number of target actions <br>
/// It is an optional parameter
DEFINE_int32(a_top, -1, target_actions_num_message);

/// @brief Flag to enable image cropping during database creation<br>
/// It is an optional parameter
DEFINE_bool(crop_gallery, false, crop_gallery_message);

/// @brief Define probability threshold for face detections during registration<br>
/// It is an optional parameter
DEFINE_double(t_reg_fd, 0.9, face_threshold_registration_output_message);

/// @brief Minimum input image width & heigh for sucessful face registration<br>
/// It is an optional parameter
DEFINE_int32(min_size_fr, 128, min_size_fr_reg_output_message);

/// @brief Output file name to save per-person action detections in.
/// It is an optional parameter
DEFINE_string(al, "", act_det_output_message);

/// @brief Number of frames to smooth actions<br>
/// It is an optional parameter
DEFINE_int32(ss_t, -1, tracker_smooth_size_message);

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
}
