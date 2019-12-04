// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>

static const char help_message[] = "Print a usage message";

static const char model_path_message[] = "Required. Path to an .xml file with a trained model.";

static const char target_device_message[] = "Optional. Specify the target device for a network (the list of available devices is shown below). " \
"Default value is CPU. Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. " \
"The demo looks for a suitable plugin for a specified device.";

static const char performance_counter_message[] = "Optional. Enable per-layer performance report";

static const char custom_cldnn_message[] = "Required for GPU custom kernels. "\
"Absolute path to an .xml file with the kernels descriptions";

static const char custom_cpu_library_message[] = "Required for CPU custom layers. " \
"Absolute path to a shared library with the kernels implementations";

static const char no_show_processed_video[] = "Optional. Do not show processed video.";

static const char num_cameras[] = "Optional. Maximum number of processed camera inputs (web cameras)";

static const char batch_size[] = "Optional. Batch size for processing (the number of frames processed per infer request)";

static const char num_infer_requests[] = "Optional. Number of infer requests";

static const char input_queue_size[] = "Optional. Frame queue size for input channels";

static const char fps_sampling_period[] = "Optional. FPS measurement sampling period between timepoints in msec";

static const char num_sampling_periods[] = "Optional. Number of sampling periods";

static const char show_statistics[] = "Optional. Enable statistics report";

static const char duplication_channel_number[] = "Optional. Enable and specify the number of channels additionally copied from real sources";

static const char real_input_fps[] = "Optional. Disable input frames caching, for maximum throughput pipeline";

static const char input_video[] = "Optional. Specify full path to input video files";

/// \brief Define a flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// \brief Define a parameter for a model file <br>
/// It is a required parameter
DEFINE_string(m, "", model_path_message);

/// \brief Define a target device parameter for a model <br>
DEFINE_string(d, "CPU", target_device_message);

/// \brief Define a flag to enable per-layer performance report <br>
DEFINE_bool(pc, false, performance_counter_message);

/// @brief GPU custom kernels path <br>
/// Default is ./lib
DEFINE_string(c, "", custom_cldnn_message);

/// @brief Absolute path to CPU library with user layers <br>
/// It is a optional parameter
DEFINE_string(l, "", custom_cpu_library_message);

/// \brief Flag to disable showing processed video<br>
/// It is an optional parameter
DEFINE_bool(no_show, false, no_show_processed_video);

/// \brief Flag to specify the number of expected input channels<br>
/// It is an optional parameter
DEFINE_uint32(nc, 0, num_cameras);

/// \brief Flag to specify batch size<br>
/// It is an optional parameter
DEFINE_uint32(bs, 1, batch_size);

/// \brief Flag to specify the number of infer requests<br>
/// It is an optional parameter
DEFINE_uint32(nireq, 5, num_infer_requests);

/// \brief Flag to specify the number of expected input channels<br>
/// It is an optional parameter
DEFINE_uint32(n_iqs, 5, input_queue_size);

/// \brief Flag to specify FPS measurement sampling period<br>
/// It is an optional parameter
DEFINE_uint32(fps_sp, 1000, fps_sampling_period);

/// \brief Flag to specify the number of sampling periods<br>
/// It is an optional parameter
DEFINE_uint32(n_sp, 10, num_sampling_periods);

/// \brief Flag to enable statisics output<br>
/// It is an optional parameter
DEFINE_bool(show_stats, false, show_statistics);

/// \brief Flag to enable and specify the number of channels additionally copied from real sources<br>
/// It is an optional parameter
DEFINE_uint32(duplicate_num, 0, duplication_channel_number);

/// \brief Flag to enable real input FPS<br>
/// It is an optional parameter
DEFINE_bool(real_input_fps, false, real_input_fps);

/// \brief Define parameter for input video files <br>
/// It is a optional parameter
DEFINE_string(i, "", input_video);
