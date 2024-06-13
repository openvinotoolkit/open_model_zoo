// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <gflags/gflags.h>
#include <utils/default_flags.hpp>

DEFINE_INPUT_FLAGS
DEFINE_OUTPUT_FLAGS

static const char help_message[] = "Print a usage message.";
static const char det_model_message[] = "Required. Path to an .xml file with a detection trained model.";
static const char cls_model_message[] = "Required. Path to an .xml file with a recognition trained model.";
static const char det_labels_message[] = "Required. Path to detection model labels file.\n"
              "                Labels file should be of the following format:\n"
              "                cat\n"
              "                dog\n";

static const char cls_labels_message[] = "Required. Path to classification model labels file.\n"
              "                Labels file should be of the following format:\n"
              "                cat\n"
              "                dog\n";

static const char num_classes_message[] = "";

static const char detection_device_message[] = "Optional. Target device for detection network (the list of available devices is shown below). "
                                               "The demo will look for a suitable plugin for a specified device. Default value is \"CPU\".";
static const char classifier_device_message[] = "Optional. Target device for recognition network (the list of available devices is shown below). "
                                                 "The demo will look for a suitable plugin for a specified device. Default value is \"CPU\".";

static const char det_nireq_message[] = "Optional. Number of infer requests. If this option is omitted, number of infer requests is determined automatically.";
static const char det_num_threads_message[] = "Optional. Number of threads.";
static const char det_num_streams_message[] = "Optional. Number of streams to use for inference on the CPU or/and GPU in "
"throughput mode (for HETERO and MULTI device cases use format "
"<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>)";

static const char cls_nireq_message[] = "Optional. Number of infer requests. If this option is omitted, number of infer requests is determined automatically.";
static const char cls_num_threads_message[] = "Optional. Number of threads.";
static const char cls_num_streams_message[] = "Optional. Number of streams to use for inference on the CPU or/and GPU in "
"throughput mode (for HETERO and MULTI device cases use format "
"<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>)";

static const char parser_message[] = "Optional. Parser kind for detector. Possible values: ssd, yolo";
static const char no_show_message[] = "Optional. Don't show output.";
static const char utilization_monitors_message[] = "Optional. List of monitors to show initially.";

DEFINE_bool(h, false, help_message);
DEFINE_string(dm, "", det_model_message);
DEFINE_string(cm, "", cls_model_message);

DEFINE_string(det_labels, "", det_labels_message);
DEFINE_string(cls_labels, "", cls_labels_message);

DEFINE_uint32(num_classes, 0, num_classes_message);

DEFINE_uint32(det_nireq, 1, det_nireq_message);
DEFINE_uint32(det_nthreads, 0, det_num_threads_message);
DEFINE_string(det_nstreams, "", det_num_streams_message);

DEFINE_uint32(cls_nireq, 1, cls_nireq_message);
DEFINE_uint32(cls_nthreads, 0, cls_num_threads_message);
DEFINE_string(cls_nstreams, "", cls_num_streams_message);

DEFINE_string(ddm, "CPU", detection_device_message);
DEFINE_string(dcm, "CPU", classifier_device_message);
DEFINE_string(parser, "ssd", parser_message);
DEFINE_bool(no_show, false, no_show_message);
DEFINE_string(u, "", utilization_monitors_message);

/**
* \brief This function shows a help message
*/

static void showUsage() {
    std::cout << std::endl;
    std::cout << "background_subtraction_demo_gapi [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                         " << help_message << std::endl;
    std::cout << "    -i                         " << input_message << std::endl;
    std::cout << "    -loop                      " << loop_message << std::endl;
    std::cout << "    -o \"<path>\"                " << output_message << std::endl;
    std::cout << "    -limit \"<num>\"             " << limit_message << std::endl;
    std::cout << "    -dm \"<path>\"                " << det_model_message << std::endl;
    std::cout << "    -cm \"<path>\"                " << cls_model_message << std::endl;
    std::cout << "    -det_labels \"<path>\"                " << det_labels_message << std::endl;
    std::cout << "    -cls_labels \"<path>\"                " << cls_labels_message << std::endl;
    std::cout << "    -num_classes \"<num>\"                " << num_classes_message << std::endl;
    std::cout << "    -ddm \"<device>\"              " << detection_device_message << std::endl;
    std::cout << "    -cdm \"<device>\"              " << classifier_device_message << std::endl;
    std::cout << "    -det_nireq \"<integer>\"         " << det_nireq_message << std::endl;
    std::cout << "    -det_nthreads \"<integer>\"      " << det_num_threads_message << std::endl;
    std::cout << "    -det_nstreams                  " << det_num_streams_message << std::endl;
    std::cout << "    -cls_nireq \"<integer>\"         " << cls_nireq_message << std::endl;
    std::cout << "    -cls_nthreads \"<integer>\"      " << cls_num_threads_message << std::endl;
    std::cout << "    -cls_nstreams                  " << cls_num_streams_message << std::endl;
    std::cout << "    -no_show                   " << no_show_message << std::endl;
    std::cout << "    -u                         " << utilization_monitors_message << std::endl;
}
