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
static const char video_message[] = "Required. Path to a video or image file. Default value is \"cam\" to work with camera.";

/// @brief message for model argument
static const char person_vehicle_bike_detection_model_message[] = "Required. Path to the Person/Vehicle/Bike Detection Crossroad model (.xml) file.";
static const char person_attribs_model_message[] = "Optional. Path to the Person Attributes Recognition Crossroad model (.xml) file.";
static const char person_reid_model_message[] = "Optional. Path to the Person Reidentification Retail model (.xml) file.";

/// @brief message for assigning Person/Vehicle/Bike detection inference to device
static const char target_device_message[] = "Optional. Specify the target device for Person/Vehicle/Bike Detection. " \
                                            "The list of available devices is shown below. Default value is CPU. " \
                                            "Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. " \
                                            "The application looks for a suitable plugin for the specified device.";

/// @brief message for assigning Person attributes recognition inference to device
static const char target_device_message_person_attribs[] = "Optional. Specify the target device for Person Attributes Recognition. "\
                                                            "The list of available devices is shown below. Default value is CPU. " \
                                                            "Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. " \
                                                            "The application looks for a suitable plugin for the specified device.";

/// @brief message for assigning Person Reidentification retail inference to device
static const char target_device_message_person_reid[] = "Optional. Specify the target device for Person Reidentification Retail. "\
                                                        "The list of available devices is shown below. Default value is CPU. " \
                                                        "Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. " \
                                                        "The application looks for a suitable plugin for the specified device.";

/// @brief message for performance counters
static const char performance_counter_message[] = "Optional. Enables per-layer performance statistics.";

/// @brief message for clDNN custom kernels desc
static const char custom_cldnn_message[] = "Optional. For clDNN (GPU)-targeted custom kernels, if any. "\
"Absolute path to the xml file with the kernels desc.";

/// @brief message for user library argument
static const char custom_cpu_library_message[] = "Optional. For MKLDNN (CPU)-targeted custom layers, if any. " \
"Absolute path to a shared library with the kernels impl.";

/// @brief message for probability threshold argument for person/vehicle/bike crossroad detections
static const char threshold_output_message[] = "Optional. Probability threshold for person/vehicle/bike crossroad detections.";

/// @brief message for probability threshold argument for person/vehicle/bike crossroad detections
static const char threshold_output_message_person_reid[] = "Optional. Cosine similarity threshold between two vectors for person reidentification.";

/// @brief message raw output flag
static const char raw_output_message[] = "Optional. Output Inference results as raw values.";

/// @brief message no show processed video
static const char no_show_processed_video[] = "Optional. No show processed video.";

/// @brief message resizable input flag
static const char input_resizable_message[] = "Optional. Enables resizable input with support of ROI crop & auto resize.";


/// @brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// @brief Define parameter for set image file <br>
/// It is a required parameter
DEFINE_string(i, "cam", video_message);

/// @brief Define parameter for vehicle detection  model file <br>
/// It is a required parameter
DEFINE_string(m, "", person_vehicle_bike_detection_model_message);

/// @brief Define parameter for vehicle attributes model file <br>
/// It is a required parameter
DEFINE_string(m_pa, "", person_attribs_model_message);

/// @brief Define parameter for vehicle detection  model file <br>
/// It is a required parameter
DEFINE_string(m_reid, "", person_reid_model_message);

/// @brief device the target device for vehicle detection infer on <br>
DEFINE_string(d, "CPU", target_device_message);

/// @brief device the target device for age gender detection on <br>
DEFINE_string(d_pa, "CPU", target_device_message_person_attribs);

/// @brief device the target device for head pose detection on <br>
DEFINE_string(d_reid, "CPU", target_device_message_person_reid);

/// @brief Enable per-layer performance report
DEFINE_bool(pc, false, performance_counter_message);

/// @brief clDNN custom kernels path <br>
/// Default is ./lib
DEFINE_string(c, "", custom_cldnn_message);

/// @brief Absolute path to CPU library with user layers <br>
/// It is a optional parameter
DEFINE_string(l, "", custom_cpu_library_message);

/// @brief Flag to output raw scoring results<br>
/// It is an optional parameter
DEFINE_bool(r, false, raw_output_message);

/// @brief Define probability threshold for person/vehicle/bike crossroad detections <br>
/// It is an optional parameter
DEFINE_double(t, 0.5, threshold_output_message);

/// @brief Define probability threshold for person/vehicle/bike crossroad detections <br>
/// It is an optional parameter
DEFINE_double(t_reid, 0.7, threshold_output_message_person_reid);

/// @brief Flag to disable processed video showing<br>
/// It is an optional parameter
DEFINE_bool(no_show, false, no_show_processed_video);

/// \brief Enables resizable input<br>
/// It is an optional parameter
DEFINE_bool(auto_resize, false, input_resizable_message);


/**
* @brief This function show a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "crossroad_camera_demo [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                           " << help_message << std::endl;
    std::cout << "    -i \"<path>\"                  " << video_message << std::endl;
    std::cout << "    -m \"<path>\"                  " << person_vehicle_bike_detection_model_message<< std::endl;
    std::cout << "    -m_pa \"<path>\"               " << person_attribs_model_message << std::endl;
    std::cout << "    -m_reid \"<path>\"             " << person_reid_model_message << std::endl;
    std::cout << "      -l \"<absolute_path>\"       " << custom_cpu_library_message << std::endl;
    std::cout << "          Or" << std::endl;
    std::cout << "      -c \"<absolute_path>\"       " << custom_cldnn_message << std::endl;
    std::cout << "    -d \"<device>\"                " << target_device_message << std::endl;
    std::cout << "    -d_pa \"<device>\"             " << target_device_message_person_attribs << std::endl;
    std::cout << "    -d_reid \"<device>\"           " << target_device_message_person_reid << std::endl;
    std::cout << "    -pc                          " << performance_counter_message << std::endl;
    std::cout << "    -r                           " << raw_output_message << std::endl;
    std::cout << "    -t                           " << threshold_output_message << std::endl;
    std::cout << "    -t_reid                      " << threshold_output_message_person_reid << std::endl;
    std::cout << "    -no_show                     " << no_show_processed_video << std::endl;
    std::cout << "    -auto_resize                 " << input_resizable_message << std::endl;
}
