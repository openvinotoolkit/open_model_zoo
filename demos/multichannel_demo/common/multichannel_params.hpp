// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

/// @brief message for model argument
static const char face_detection_model_message[] = "Required. Path to an .xml file with a trained face detection model.";

/// @brief message for plugin argument
static const char plugin_message[] = "Plugin name. For example CPU. If this parameter is set, " \
"the demo will look for this plugin only.";

/// @brief message for assigning face detection calculation to device
static const char target_device_message[] = "Optional. Specify the target device for Face Detection (CPU, GPU, FPGA, HDDL or MYRIAD). " \
"The demo will look for a suitable plugin for a specified device.";

/// @brief message for performance counters
static const char performance_counter_message[] = "Optional. Enables per-layer performance report.";

/// @brief message for clDNN custom kernels desc
static const char custom_cldnn_message[] = "Optional. Required for GPU custom kernels. "\
"Absolute path to the .xml file with the kernels descriptions.";

/// @brief message for user library argument
static const char custom_cpu_library_message[] = "Optional. Required for CPU custom layers. " \
"Absolute path to a shared library with the kernels implementations.";

/// @brief message no show processed video
static const char no_show_processed_video[] = "Optional. Do not show processed video.";

/// @brief message for number of camera inputs
static const char num_cameras[] = "Optional. Maximum number of processed camera inputs (web cameras)";

/// @brief message for batch_size
static const char batch_size[] = "Optional. Processing batch size, number of frames processed per infer request";

/// @brief message for number of infer requests
static const char num_infer_requests[] = "Optional. Number of infer requests";

/// @brief message for inputs queue size
static const char input_queue_size[] = "Optional. Frame queue size for input channels";

/// @brief message for FPS measurement sampling period
static const char fps_sampling_period[] = "Optional. FPS measurement sampling period. Duration between timepoints, msec";

/// @brief message for FPS measurement sampling period
static const char num_sampling_periods[] = "Optional. Number of sampling periods";

/// @brief message for enabling statistics output
static const char show_statistics[] = "Optional. Enable statictics output";

/// @brief message for enabling channel duplication
static const char duplication_channel_number[] = "Optional. Enable and specify number of channel additionally copied from real sources";

/// @brief message for enabling real input FPS
static const char real_input_fps[] = "Optional. Disable input frames caching, for maximum throughput pipeline";

/// @brief message for enabling input video
static const char input_video[] = "Optional. Specify full path to input video files";

/// \brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// \brief Define parameter for face detection model file <br>
/// It is a required parameter
DEFINE_string(m, "", face_detection_model_message);

/// \brief target device for face detection <br>
DEFINE_string(d, "CPU", target_device_message);

/// \brief enable per-layer performance report <br>
DEFINE_bool(pc, false, performance_counter_message);

/// @brief clDNN custom kernels path <br>
/// Default is ./lib
DEFINE_string(c, "", custom_cldnn_message);

/// @brief Absolute path to CPU library with user layers <br>
/// It is a optional parameter
DEFINE_string(l, "", custom_cpu_library_message);

/// \brief Flag to disable processed video showing<br>
/// It is an optional parameter
DEFINE_bool(no_show, false, no_show_processed_video);

/// \brief Flag to specify number of expected input channels<br>
/// It is an optional parameter
DEFINE_uint32(nc, 0, num_cameras);

/// \brief Flag to specify batch size<br>
/// It is an optional parameter
DEFINE_uint32(bs, 1, batch_size);

/// \brief Flag to specify number of infer requests<br>
/// It is an optional parameter
DEFINE_uint32(n_ir, 5, num_infer_requests);

/// \brief Flag to specify number of expected input channels<br>
/// It is an optional parameter
DEFINE_uint32(n_iqs, 5, input_queue_size);

/// \brief Flag to specify FPS measurement sampling period<br>
/// It is an optional parameter
DEFINE_uint32(fps_sp, 1000, fps_sampling_period);

/// \brief Flag to specify number of sampling periods<br>
/// It is an optional parameter
DEFINE_uint32(n_sp, 10, num_sampling_periods);

/// \brief Flag to enable statisics output<br>
/// It is an optional parameter
DEFINE_bool(show_stats, false, show_statistics);

/// \brief Flag to enable statisics output<br>
/// It is an optional parameter
DEFINE_uint32(duplicate_num, 0, duplication_channel_number);

/// \brief Flag to enable statisics output<br>
/// It is an optional parameter
DEFINE_bool(real_input_fps, false, real_input_fps);

/// \brief Define parameter for input video files <br>
/// It is a optional parameter
DEFINE_string(i, "", input_video);
