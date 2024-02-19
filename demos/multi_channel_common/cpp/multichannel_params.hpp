// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gflags/gflags.h>

static const char help_message[] = "Print a usage message";
DEFINE_bool(h, false, help_message);

static const char input_message[] = "A comma separated list of inputs to process. Each input must be a "
    "single image, a folder of images or anything that cv::VideoCapture can process.";
DEFINE_string(i, "", input_message);

static const char loop_message[] = "Enable reading the inputs in a loop.";
DEFINE_bool(loop, false, loop_message);

static const char duplication_channel_number_message[] = "Multiply the inputs by the given factor."
    " For example, if only one input is provided, but -duplicate_num is set to 2, the demo will split real input across channels,"
    " by interleaving frames between channels.";
DEFINE_uint32(duplicate_num, 1, duplication_channel_number_message);

static const char model_path_message[] = "Path to an .xml file with a trained model.";
DEFINE_string(m, "", model_path_message);

static const char target_device_message[] =
    "Specify a target device to infer on (the list of available devices is shown below). "
    "Default value is CPU. Use \"-d HETERO:<comma-separated_devices_list>\" format to specify "
    "HETERO plugin. "
    "Use \"-d MULTI:<comma-separated_devices_list>\" format to specify MULTI plugin. "
    "The application looks for a suitable plugin for the specified device.";
DEFINE_string(d, "CPU", target_device_message);

static const char no_show_message[] = "Don't show output.";
DEFINE_bool(no_show, false, no_show_message);

static const char batch_size[] = "Batch size for processing (the number of frames processed per infer request)";
DEFINE_uint32(bs, 1, batch_size);

static const char input_queue_size[] = "Frame queue size for input channels";
DEFINE_uint32(n_iqs, 5, input_queue_size);

static const char fps_sampling_period[] = "FPS measurement sampling period between timepoints in msec";
DEFINE_uint32(fps_sp, 1000, fps_sampling_period);

static const char num_sampling_periods[] = "Number of sampling periods";
DEFINE_uint32(n_sp, 10, num_sampling_periods);

static const char show_statistics[] = "Enable statistics report";
DEFINE_bool(show_stats, false, show_statistics);

static const char real_input_fps[] = "Disable input frames caching, for maximum throughput pipeline";
DEFINE_bool(real_input_fps, false, real_input_fps);

static const char utilization_monitors_message[] = "List of monitors to show initially.";
DEFINE_string(u, "", utilization_monitors_message);
