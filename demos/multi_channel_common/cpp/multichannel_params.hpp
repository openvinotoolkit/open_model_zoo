// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>

static const char help_message[] = "Print a usage message";
static const char input_message[] = "Required. A comma separated list of inputs to process. Each input must be a "
    "single image, a folder of images or anything that cv::VideoCapture can process.";
static const char loop_message[] = "Optional. Enable reading the inputs in a loop.";
static const char duplication_channel_number_message[] = "Optional. Multiply the inputs by the given factor. For "
    "example, if only one input is provided, but -ni is set to 2, the demo uses half of images from the input as it was"
    " the first input and another half goes as the second input.";
static const char model_path_message[] = "Required. Path to an .xml file with a trained model.";
static const char target_device_message[] = "Optional. Specify the target device for a network (the list of available devices is shown below). "
                                            "Default value is CPU. Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. "
                                            "The demo looks for a suitable plugin for a specified device.";
static const char performance_counter_message[] = "Optional. Enable per-layer performance report";
static const char custom_cldnn_message[] = "Required for GPU custom kernels. "
                                           "Absolute path to an .xml file with the kernels descriptions";
static const char custom_cpu_library_message[] = "Required for CPU custom layers. "
                                                 "Absolute path to a shared library with the kernels implementations";
static const char no_show_processed_video[] = "Optional. Do not show processed video.";
static const char batch_size[] = "Optional. Batch size for processing (the number of frames processed per infer request)";
static const char num_infer_requests[] = "Optional. Number of infer requests";
static const char input_queue_size[] = "Optional. Frame queue size for input channels";
static const char fps_sampling_period[] = "Optional. FPS measurement sampling period between timepoints in msec";
static const char num_sampling_periods[] = "Optional. Number of sampling periods";
static const char show_statistics[] = "Optional. Enable statistics report";
static const char real_input_fps[] = "Optional. Disable input frames caching, for maximum throughput pipeline";
static const char utilization_monitors_message[] = "Optional. List of monitors to show initially.";

DEFINE_bool(h, false, help_message);
DEFINE_string(i, "", input_message);
DEFINE_bool(loop, false, loop_message);
DEFINE_uint32(duplicate_num, 1, duplication_channel_number_message);
DEFINE_string(m, "", model_path_message);
DEFINE_string(d, "CPU", target_device_message);
DEFINE_bool(pc, false, performance_counter_message);
DEFINE_string(c, "", custom_cldnn_message);
DEFINE_string(l, "", custom_cpu_library_message);
DEFINE_bool(no_show, false, no_show_processed_video);
DEFINE_uint32(bs, 1, batch_size);
DEFINE_uint32(nireq, 5, num_infer_requests);
DEFINE_uint32(n_iqs, 5, input_queue_size);
DEFINE_uint32(fps_sp, 1000, fps_sampling_period);
DEFINE_uint32(n_sp, 10, num_sampling_periods);
DEFINE_bool(show_stats, false, show_statistics);
DEFINE_bool(real_input_fps, false, real_input_fps);
DEFINE_string(u, "", utilization_monitors_message);
