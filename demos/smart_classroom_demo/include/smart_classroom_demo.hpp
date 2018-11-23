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

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>

#ifdef _WIN32
#include <os/windows/w_dirent.h>
#else
#include <dirent.h>
#endif

/// @brief message for help argument
static const char help_message[] = "Print a usage message.";

/// @brief message for images argument
static const char video_message[] = "Required. Path to a video or image file. Default value is \"cam\" to work with camera.";

/// @brief message for model argument
static const char person_action_detection_model_message[] = "Required. Path to the Person/Action Detection Retail model (.xml) file.";
static const char face_detection_model_message[] = "Required. Path to the Face Detection Retail model (.xml) file.";
static const char facial_landmarks_model_message[] = "Required. Path to the Facial Landmarks Regression Retail model (.xml) file.";
static const char face_reid_model_message[] = "Required. Path to the Face Reidentification Retail model (.xml) file.";

/// @brief message for assigning Person/Action detection inference to device
static const char target_device_message_action_detection[] = "Optional. Specify the target device for Person/Action Detection Retail "\
                                            "(CPU, GPU, FPGA, MYRIAD, or HETERO). ";

/// @brief message for assigning Face Detection inference to device
static const char target_device_message_face_detection[] = "Optional. Specify the target device for Face Detection Retail "\
                                                           "(CPU, GPU, FPGA, MYRIAD, or HETERO).";

/// @brief message for assigning Landmarks Regression retail inference to device
static const char target_device_message_landmarks_regression[] = "Optional. Specify the target device for Landmarks Regression Retail "\
                                                        "(CPU, GPU, FPGA, MYRIAD, or HETERO). ";

/// @brief message for assigning Face Reidentification retail inference to device
static const char target_device_message_face_reid[] = "Optional. Specify the target device for Face Reidentification Retail "\
                                                        "(CPU, GPU, FPGA, MYRIAD, or HETERO). ";

/// @brief message for performance counters
static const char performance_counter_message[] = "Optional. Enables per-layer performance statistics.";

/// @brief message for clDNN custom kernels desc
static const char custom_cldnn_message[] = "Optional. For clDNN (GPU)-targeted custom kernels, if any. "\
"Absolute path to the xml file with the kernels desc.";

/// @brief message for user library argument
static const char custom_cpu_library_message[] = "Optional. For MKLDNN (CPU)-targeted custom layers, if any. " \
"Absolute path to a shared library with the kernels impl.";

/// @brief message for probability threshold argument for face detections
static const char face_threshold_output_message[] = "Optional. Probability threshold for face detections.";

/// @brief message for probability threshold argument for persons/actions detections
static const char person_threshold_output_message[] = "Optional. Probability threshold for persons/actions detections.";

/// @brief message for cosine distance threshold for face reidentification
static const char threshold_output_message_face_reid[] = "Optional. Cosine distance threshold between two vectors for face reidentification.";

/// @brief message for faces gallery path
static const char reid_gallery_path_message[] = "Optional. Path to a faces gallery in json format.";

/// @brief message for output video path
static const char output_video_message[] = "Optional. File to write output video with visualization to.";

/// @brief message raw output flag
static const char raw_output_message[] = "Optional. Output Inference results as raw values.";

/// @brief message no show processed video
static const char no_show_processed_video[] = "Optional. No show processed video.";

/// @brief message input image height for face detector
static const char input_image_height_output_message[] = "Optional. Input image height for face detector.";

/// @brief message input image width for face detector
static const char input_image_width_output_message[] = "Optional. Input image width for face detector.";

/// @brief message expand ratio for bbox
static const char expand_ratio_output_message[] = "Optional. Expand ratio for bbox before face recognition.";

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

/// @brief Flag to output raw pipeline results<br>
/// It is an optional parameter
DEFINE_bool(r, false, raw_output_message);

/// @brief Define probability threshold for person/action detections <br>
/// It is an optional parameter
DEFINE_double(t_act, 0.65, person_threshold_output_message);

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

/**
* @brief This function show a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "smart_classroom_demo [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                             " << help_message << std::endl;
    std::cout << "    -i \"<path>\"                  " << video_message << std::endl;
    std::cout << "    -m_act \"<path>\"              " << person_action_detection_model_message << std::endl;
    std::cout << "    -m_fd \"<path>\"               " << face_detection_model_message << std::endl;
    std::cout << "    -m_lm \"<path>\"               " << facial_landmarks_model_message << std::endl;
    std::cout << "    -m_reid \"<path>\"             " << face_reid_model_message << std::endl;
    std::cout << "    -l \"<absolute_path>\"         " << custom_cpu_library_message << std::endl;
    std::cout << "          Or" << std::endl;
    std::cout << "    -c \"<absolute_path>\"         " << custom_cldnn_message << std::endl;
    std::cout << "    -d_act \"<device>\"            " << target_device_message_action_detection << std::endl;
    std::cout << "    -d_fd \"<device>\"             " << target_device_message_face_detection << std::endl;
    std::cout << "    -d_lm \"<device>\"             " << target_device_message_landmarks_regression << std::endl;
    std::cout << "    -d_reid \"<device>\"           " << target_device_message_face_reid << std::endl;
    std::cout << "    -out_v  \"<path>\"             " << output_video_message << std::endl;
    std::cout << "    -pc                            " << performance_counter_message << std::endl;
    std::cout << "    -r                             " << raw_output_message << std::endl;
    std::cout << "    -t_act                         " << person_threshold_output_message << std::endl;
    std::cout << "    -t_fd                          " << face_threshold_output_message << std::endl;
    std::cout << "    -inh_fd                        " << input_image_height_output_message << std::endl;
    std::cout << "    -inw_fd                        " << input_image_width_output_message << std::endl;
    std::cout << "    -exp_r_fd                      " << expand_ratio_output_message << std::endl;
    std::cout << "    -t_reid                        " << threshold_output_message_face_reid << std::endl;
    std::cout << "    -fg                            " << reid_gallery_path_message << std::endl;
    std::cout << "    -no_show                       " << no_show_processed_video << std::endl;
}
