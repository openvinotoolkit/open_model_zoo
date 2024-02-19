// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "gflags/gflags.h"
#include "utils/default_flags.hpp"

DEFINE_INPUT_FLAGS
DEFINE_OUTPUT_FLAGS

static const char help_message[] = "Print a usage message.";
static const char person_vehicle_bike_detection_model_message[] = "Required. Path to the Person/Vehicle/Bike Detection Crossroad model (.xml) file.";
static const char person_attribs_model_message[] = "Optional. Path to the Person Attributes Recognition Crossroad model (.xml) file.";
static const char person_reid_model_message[] = "Optional. Path to the Person Reidentification Retail model (.xml) file.";
static const char target_device_message[] = "Optional. Specify the target device for Person/Vehicle/Bike Detection. "
                                            "The list of available devices is shown below. Default value is CPU. "
                                            "Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. "
                                            "The application looks for a suitable plugin for the specified device.";
static const char target_device_message_person_attribs[] = "Optional. Specify the target device for Person Attributes Recognition. "
                                                            "The list of available devices is shown below. Default value is CPU. "
                                                            "Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. "
                                                            "The application looks for a suitable plugin for the specified device.";
static const char target_device_message_person_reid[] = "Optional. Specify the target device for Person Reidentification Retail. "
                                                        "The list of available devices is shown below. Default value is CPU. "
                                                        "Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. "
                                                        "The application looks for a suitable plugin for the specified device.";
static const char threshold_output_message[] = "Optional. Probability threshold for person/vehicle/bike crossroad detections.";
static const char threshold_output_message_person_reid[] = "Optional. Cosine similarity threshold between two vectors for person reidentification.";
static const char raw_output_message[] = "Optional. Output Inference results as raw values.";
static const char no_show_message[] = "Optional. Don't show output.";
static const char input_resizable_message[] = "Optional. Enables resizable input with support of ROI crop & auto resize.";
static const char utilization_monitors_message[] = "Optional. List of monitors to show initially.";
static const char person_label_message[] = "Optional. The integer index of the objects' category corresponding to persons "
                                           "(as it is returned from the detection network, may vary from one network to another). "
                                           "The default value is 1.";


DEFINE_bool(h, false, help_message);
DEFINE_string(m, "", person_vehicle_bike_detection_model_message);
DEFINE_string(m_pa, "", person_attribs_model_message);
DEFINE_string(m_reid, "", person_reid_model_message);
DEFINE_string(d, "CPU", target_device_message);
DEFINE_string(d_pa, "CPU", target_device_message_person_attribs);
DEFINE_string(d_reid, "CPU", target_device_message_person_reid);
DEFINE_bool(r, false, raw_output_message);
DEFINE_double(t, 0.5, threshold_output_message);
DEFINE_double(t_reid, 0.7, threshold_output_message_person_reid);
DEFINE_bool(no_show, false, no_show_message);
DEFINE_bool(auto_resize, false, input_resizable_message);
DEFINE_string(u, "", utilization_monitors_message);
DEFINE_int32(person_label, 1, person_label_message);


/**
* @brief This function show a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "crossroad_camera_demo [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                           " << help_message << std::endl;
    std::cout << "    -i                           " << input_message << std::endl;
    std::cout << "    -loop                        " << loop_message << std::endl;
    std::cout << "    -o \"<path>\"                  " << output_message << std::endl;
    std::cout << "    -limit \"<num>\"               " << limit_message << std::endl;
    std::cout << "    -m \"<path>\"                  " << person_vehicle_bike_detection_model_message<< std::endl;
    std::cout << "    -m_pa \"<path>\"               " << person_attribs_model_message << std::endl;
    std::cout << "    -m_reid \"<path>\"             " << person_reid_model_message << std::endl;
    std::cout << "    -d \"<device>\"                " << target_device_message << std::endl;
    std::cout << "    -d_pa \"<device>\"             " << target_device_message_person_attribs << std::endl;
    std::cout << "    -d_reid \"<device>\"           " << target_device_message_person_reid << std::endl;
    std::cout << "    -r                           " << raw_output_message << std::endl;
    std::cout << "    -t                           " << threshold_output_message << std::endl;
    std::cout << "    -t_reid                      " << threshold_output_message_person_reid << std::endl;
    std::cout << "    -no_show                     " << no_show_message << std::endl;
    std::cout << "    -auto_resize                 " << input_resizable_message << std::endl;
    std::cout << "    -u                           " << utilization_monitors_message << std::endl;
    std::cout << "    -person_label                " << person_label_message << std::endl;
}
