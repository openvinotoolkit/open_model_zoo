// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>

/// @brief message for help argument
static const char help_message[] = "Print a usage message.";

/// @brief message for images argument
static const char video_message[] = "Required. Path to a video file or a folder with images "\
                                     "(all images should have names 0000000001.jpg, 0000000002.jpg, etc).";

/// @brief message for model arguments
static const char pedestrian_detection_model_message[] = "Required. Path to the Pedestrian Detection Retail model (.xml) file.";
static const char pedestrian_reid_model_message[] = "Required. Path to the Pedestrian Reidentification Retail model (.xml) file.";

/// @brief message for assigning Pedestrian detection inference to device
static const char target_device_detection_message[] = "Optional. Specify the target device for pedestrian detection "\
                                                      "(the list of available devices is shown below). Default value is CPU. " \
                                                      "Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin.";

/// @brief message for assigning Pedestrian Reidentification retail inference to device
static const char target_device_reid_message[] = "Optional. Specify the target device for pedestrian reidentification "\
                                                 "(the list of available devices is shown below). Default value is CPU. " \
                                                 "Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin.";

/// @brief message for performance counters
static const char performance_counter_message[] = "Optional. Enable per-layer performance statistics.";

/// @brief message for clDNN custom kernels desc
static const char custom_cldnn_message[] = "Optional. For GPU custom kernels, if any. "\
                                            "Absolute path to the .xml file with the kernels description.";

/// @brief message for user library argument
static const char custom_cpu_library_message[] = "Optional. For CPU custom layers, if any. "\
                                                  "Absolute path to a shared library with the kernels implementation.";

/// @brief message raw output flag
static const char raw_output_message[] = "Optional. Output pedestrian tracking results in a raw format "\
                                          "(compatible with MOTChallenge format).";

/// @brief message no show processed video
static const char no_show_processed_video[] = "Optional. Do not show processed video.";

/// @brief message for delay between frames
static const char delay_message[] = "Optional. Delay between frames used for visualization. "\
                                     "If negative, the visualization is turned off (like with the option 'no_show'). "\
                                     "If zero, the visualization is made frame-by-frame.";

/// @brief message for output log
static const char output_log_message[] = "Optional. The file name to write output log file with results of pedestrian tracking. "\
                                          "The format of the log file is compatible with MOTChallenge format.";

/// @brief message for the first frame
static const char first_frame_message[] = "Optional. The index of the first frame of video sequence to process. "\
                                           "This has effect only if it is positive and the source video sequence is an image folder.";
/// @brief message for the last frame
static const char last_frame_message[] = "Optional. The index of the last frame of video sequence to process. "\
                                          "This has effect only if it is positive and the source video sequence is an image folder.";


/// @brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// @brief Define parameter for set image file <br>
/// It is a required parameter
DEFINE_string(i, "", video_message);

/// @brief Define parameter for pedestrian detection model file <br>
/// It is a required parameter
DEFINE_string(m_det, "", pedestrian_detection_model_message);

/// @brief Define parameter for pedestrian reidentification model file <br>
/// It is a required parameter
DEFINE_string(m_reid, "", pedestrian_reid_model_message);

/// @brief device the target device for pedestrian detection infer on <br>
DEFINE_string(d_det, "CPU", target_device_detection_message);

/// @brief device the target device for pedestrian reidentification infer on <br>
DEFINE_string(d_reid, "CPU", target_device_reid_message);

/// @brief Enable per-layer performance report
DEFINE_bool(pc, false, performance_counter_message);

/// @brief clDNN custom kernels path <br>
/// Default is ./lib
DEFINE_string(c, "", custom_cldnn_message);

/// @brief Absolute path to CPU library with user layers <br>
/// It is a optional parameter
DEFINE_string(l, "", custom_cpu_library_message);

/// \brief Flag to output pedestrian tracking results in raw format<br>
/// It is an optional parameter
DEFINE_bool(r, false, raw_output_message);

/// @brief Flag to disable processed video showing<br>
/// It is an optional parameter
DEFINE_bool(no_show, false, no_show_processed_video);

/// @brief Define delay for visualization <br>
/// It is an optional parameter
DEFINE_int32(delay, 3, delay_message);

/// @brief Define output log path to store tracking results <br>
/// It is an optional parameter
DEFINE_string(out, "", output_log_message);

/// @brief Define the first frame to process <br>
/// It is an optional parameter
DEFINE_int32(first, -1, first_frame_message);

/// @brief Define the last frame to process <br>
/// It is an optional parameter
DEFINE_int32(last, -1, last_frame_message);


/**
 * @brief This function show a help message
 */
static void showUsage() {
    std::cout << std::endl;
    std::cout << "pedestrian_tracker_demo [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                           " << help_message << std::endl;
    std::cout << "    -i \"<path>\"                  " << video_message << std::endl;
    std::cout << "    -m_det \"<path>\"              " << pedestrian_detection_model_message << std::endl;
    std::cout << "    -m_reid \"<path>\"             " << pedestrian_reid_model_message << std::endl;
    std::cout << "    -l \"<absolute_path>\"         " << custom_cpu_library_message << std::endl;
    std::cout << "          Or" << std::endl;
    std::cout << "    -c \"<absolute_path>\"         " << custom_cldnn_message << std::endl;
    std::cout << "    -d_det \"<device>\"            " << target_device_detection_message << std::endl;
    std::cout << "    -d_reid \"<device>\"           " << target_device_reid_message << std::endl;
    std::cout << "    -r                           " << raw_output_message << std::endl;
    std::cout << "    -pc                          " << performance_counter_message << std::endl;
    std::cout << "    -no_show                     " << no_show_processed_video << std::endl;
    std::cout << "    -delay                       " << delay_message << std::endl;
    std::cout << "    -out \"<path>\"                " << output_log_message << std::endl;
    std::cout << "    -first                       " << first_frame_message << std::endl;
    std::cout << "    -last                        " << last_frame_message << std::endl;
}
