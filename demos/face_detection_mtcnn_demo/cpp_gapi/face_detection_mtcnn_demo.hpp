// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <gflags/gflags.h>
#include <utils/default_flags.hpp>

DEFINE_INPUT_FLAGS
DEFINE_OUTPUT_FLAGS

static const char help_message[] = "Print a usage message.";
static const char camera_resolution_message[] = "Optional. Set camera resolution in format WxH.";
static const char mtcnn_p_model_message[] = "Required. Path to an .xml file with a trained OpenVINO MTCNN P (Proposal) detection model.";
static const char mtcnn_r_model_message[] = "Required. Path to an .xml file with a trained OpenVINO MTCNN R (Refinement) detection model.";
static const char mtcnn_o_model_message[] = "Required. Path to an .xml file with a trained OpenVINO MTCNN O (Output) detection model.";
static const char target_device_message_p[] = "Optional. Target device for MTCNN P network. "
                                              "The demo will look for a suitable plugin for a specified device. Default value is \"CPU\".";
static const char target_device_message_r[] = "Optional. Target device for MTCNN R network. "
                                              "The demo will look for a suitable plugin for a specified device. Default value is \"CPU\".";
static const char target_device_message_o[] = "Optional. Target device for MTCNN O network. "
                                              "The demo will look for a suitable plugin for a specified device. Default value is \"CPU\".";
static const char thresh_output_message[] = "Optional. MTCNN confidence threshold. The default value is 0.7.";
static const char queue_capacity_message[] = "Optional. Streaming executor queue capacity. Calculated automaticaly if 0.";
static const char half_scale_message[] = "Optional. MTCNN P use half scale pyramid.";
static const char no_show_message[] = "Optional. Don't show output.";
static const char utilization_monitors_message[] = "Optional. List of monitors to show initially.";

DEFINE_bool(h, false, help_message);
DEFINE_string(res, "1280x720", camera_resolution_message);
DEFINE_string(m_p, "", mtcnn_p_model_message);
DEFINE_string(m_r, "", mtcnn_r_model_message);
DEFINE_string(m_o, "", mtcnn_o_model_message);
DEFINE_string(d_p, "CPU", target_device_message_p);
DEFINE_string(d_r, "CPU", target_device_message_r);
DEFINE_string(d_o, "CPU", target_device_message_o);
DEFINE_uint32(qc, 1, queue_capacity_message);
DEFINE_bool(hs, false, half_scale_message);
DEFINE_double(th, 0.7, thresh_output_message);
DEFINE_bool(no_show, false, no_show_message);
DEFINE_string(u, "", utilization_monitors_message);

/**
* \brief This function shows a help message
*/

#include <iostream>
static void showUsage() {
    std::cout << std::endl;
    std::cout << "gesture_recognition_demo_gapi [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                       " << help_message << std::endl;
    std::cout << "    -i                       " << input_message << std::endl;
    std::cout << "    -loop                    " << loop_message << std::endl;
    std::cout << "    -o \"<path>\"             " << output_message << std::endl;
    std::cout << "    -limit \"<num>\"          " << limit_message << std::endl;
    std::cout << "    -res \"<WxH>\"            " << camera_resolution_message << std::endl;
    std::cout << "    -m_p \"<path>\"           " << mtcnn_p_model_message << std::endl;
    std::cout << "    -m_r \"<path>\"           " << mtcnn_r_model_message << std::endl;
    std::cout << "    -m_o \"<path>\"           " << mtcnn_o_model_message << std::endl;
    std::cout << "    -d_p \"<device>\"         " << target_device_message_p << std::endl;
    std::cout << "    -d_r \"<device>\"         " << target_device_message_r << std::endl;
    std::cout << "    -d_o \"<device>\"         " << target_device_message_o << std::endl;
    std::cout << "    -qc                      " << queue_capacity_message << std::endl;
    std::cout << "    -hs                      " << half_scale_message << std::endl;
    std::cout << "    -no_show                 " << no_show_message << std::endl;
    std::cout << "    -th                      " << thresh_output_message << std::endl;
    std::cout << "    -u                       " << utilization_monitors_message << std::endl;
}
