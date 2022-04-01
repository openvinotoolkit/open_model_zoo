// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <limits>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include <utils/default_flags.hpp>

DEFINE_INPUT_FLAGS
DEFINE_OUTPUT_FLAGS

static const char help_message[] = "Print a usage message.";
static const char first_frame_message[] = "Optional. The index of the first frame of the input to process. "
                                          "The actual first frame captured depends on cv::VideoCapture implementation "
                                          "and may have slightly different number.";
static const char read_limit_message[] = "Optional. Read length limit before stopping or restarting reading the input.";
static const char pedestrian_detection_model_message[] =
    "Required. Path to the Pedestrian Detection Retail model (.xml) file.";
static const char pedestrian_reid_model_message[] =
    "Required. Path to the Pedestrian Reidentification Retail model (.xml) file.";
static const char target_device_detection_message[] =
    "Optional. Specify the target device for pedestrian detection "
    "(the list of available devices is shown below). Default value is CPU. "
    "Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin.";
static const char target_device_reid_message[] =
    "Optional. Specify the target device for pedestrian reidentification "
    "(the list of available devices is shown below). Default value is CPU. "
    "Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin.";
static const char layout_det_model_message[] = "Optional. Specify inputs layouts."
                                               " Ex. NCHW or input0:NCHW,input1:NC in case of more than one input.";
static const char raw_output_message[] = "Optional. Output pedestrian tracking results in a raw format "
                                         "(compatible with MOTChallenge format).";
static const char no_show_message[] = "Optional. Don't show output.";
static const char delay_message[] = "Optional. Delay between frames used for visualization. "
                                    "If negative, the visualization is turned off (like with the option 'no_show'). "
                                    "If zero, the visualization is made frame-by-frame.";
static const char output_log_message[] =
    "Optional. The file name to write output log file with results of pedestrian tracking. "
    "The format of the log file is compatible with MOTChallenge format.";
static const char utilization_monitors_message[] = "Optional. List of monitors to show initially.";
static const char at_message[] = "Required. Architecture type for detector model: centernet, ssd or yolo.";
static const char thresh_output_message[] = "Optional. Probability threshold for detections.";
static const char input_resizable_message[] =
    "Optional. Enables resizable input with support of ROI crop & auto resize.";
static const char iou_thresh_output_message[] =
    "Optional. Filtering intersection over union threshold for overlapping boxes.";
static const char yolo_af_message[] = "Optional. Use advanced postprocessing/filtering algorithm for YOLO.";
static const char labels_message[] = "Optional. Path to a file with labels mapping.";
static const char nireq_message[] = "Optional. Number of infer requests for detector model. If this option is omitted, "
                                    "number of infer requests is determined automatically.";
static const char num_threads_message[] = "Optional. Number of threads for detector model.";
static const char num_streams_message[] =
    "Optional. Number of streams to use for inference on the CPU or/and GPU in "
    "throughput mode for detector model (for HETERO and MULTI device cases use format "
    "<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>)";
static const char person_label_message[] =
    "Optional. Label of class person for detector. Default -1 for tracking all objects";

DEFINE_bool(h, false, help_message);
DEFINE_uint32(first, 0, first_frame_message);
DEFINE_uint32(read_limit, static_cast<gflags::uint32>(std::numeric_limits<size_t>::max()), read_limit_message);
DEFINE_string(m_det, "", pedestrian_detection_model_message);
DEFINE_string(m_reid, "", pedestrian_reid_model_message);
DEFINE_string(d_det, "CPU", target_device_detection_message);
DEFINE_string(d_reid, "CPU", target_device_reid_message);
DEFINE_string(layout_det, "", layout_det_model_message);
DEFINE_bool(r, false, raw_output_message);
DEFINE_bool(no_show, false, no_show_message);
DEFINE_int32(delay, 3, delay_message);
DEFINE_string(out, "", output_log_message);
DEFINE_string(u, "", utilization_monitors_message);
DEFINE_string(at, "", at_message);
DEFINE_double(t, 0.5, thresh_output_message);
DEFINE_bool(auto_resize, false, input_resizable_message);
DEFINE_double(iou_t, 0.5, iou_thresh_output_message);
DEFINE_bool(yolo_af, true, yolo_af_message);
DEFINE_string(labels, "", labels_message);
DEFINE_uint32(nireq, 0, nireq_message);
DEFINE_uint32(nthreads, 0, num_threads_message);
DEFINE_string(nstreams, "", num_streams_message);
DEFINE_int32(person_label, -1, person_label_message);

/**
 * @brief This function show a help message
 */
static void showUsage() {
    std::cout << std::endl;
    std::cout << "pedestrian_tracker_demo [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                           " << help_message << std::endl;
    std::cout << "    -i                           " << input_message << std::endl;
    std::cout << "    -loop                        " << loop_message << std::endl;
    std::cout << "    -first                       " << first_frame_message << std::endl;
    std::cout << "    -read_limit                  " << read_limit_message << std::endl;
    std::cout << "    -o \"<path>\"                  " << output_message << std::endl;
    std::cout << "    -limit \"<num>\"               " << limit_message << std::endl;
    std::cout << "    -m_det \"<path>\"              " << pedestrian_detection_model_message << std::endl;
    std::cout << "    -m_reid \"<path>\"             " << pedestrian_reid_model_message << std::endl;
    std::cout << "    -d_det \"<device>\"            " << target_device_detection_message << std::endl;
    std::cout << "    -d_reid \"<device>\"           " << target_device_reid_message << std::endl;
    std::cout << "    -layout_det \"<string>\"       " << layout_det_model_message << std::endl;
    std::cout << "    -r                           " << raw_output_message << std::endl;
    std::cout << "    -no_show                     " << no_show_message << std::endl;
    std::cout << "    -delay                       " << delay_message << std::endl;
    std::cout << "    -out \"<path>\"                " << output_log_message << std::endl;
    std::cout << "    -u                           " << utilization_monitors_message << std::endl;
    std::cout << "    -at \"<type>\"              " << at_message << std::endl;
    std::cout << "    -t                          " << thresh_output_message << std::endl;
    std::cout << "    -auto_resize                " << input_resizable_message << std::endl;
    std::cout << "    -iou_t                      " << iou_thresh_output_message << std::endl;
    std::cout << "    -yolo_af                    " << yolo_af_message << std::endl;
    std::cout << "    -labels \"<path>\"          " << labels_message << std::endl;
    std::cout << "    -nireq \"<integer>\"        " << nireq_message << std::endl;
    std::cout << "    -nstreams                   " << num_streams_message << std::endl;
    std::cout << "    -nthreads \"<integer>\"     " << num_threads_message << std::endl;
    std::cout << "    -person_label               " << person_label_message << std::endl;
}
