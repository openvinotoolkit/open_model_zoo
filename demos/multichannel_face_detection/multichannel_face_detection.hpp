/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/
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
static const char plugin_message[] = "Plugin name. For example MKLDNNPlugin. If this parameter is pointed, " \
"the demo will look for this plugin only.";

/// @brief message for assigning face detection calculation to device
static const char target_device_message[] = "Specify the target device for Face Detection (CPU, GPU, FPGA, or MYRIAD). " \
"The demo will look for a suitable plugin for a specified device.";

/// @brief message for performance counters
static const char performance_counter_message[] = "Enables per-layer performance report.";

/// @brief message for clDNN custom kernels desc
static const char custom_cldnn_message[] = "Required for clDNN (GPU)-targeted custom kernels. "\
"Absolute path to the xml file with the kernels desc.";

/// @brief message for user library argument
static const char custom_cpu_library_message[] = "Required for MKLDNN (CPU)-targeted custom layers. " \
"Absolute path to a shared library with the kernels impl.";

/// @brief message for probability threshold argument
static const char thresh_output_message[] = "Probability threshold for detections.";

/// @brief message no show processed video
static const char no_show_processed_video[] = "No show processed video.";

/// @brief message for number of camera inputs
static const char num_cameras[] = "Maximum number of processed camera inputs (web cams)";

/// @brief message for batch_size
static const char batch_size[] = "Processing batch size, number of frames processed per infer request";

/// @brief message for number of infer requests
static const char num_infer_requests[] = "Number of infer requests";

/// @brief message for inputs queue size
static const char input_queue_size[] = "Frame queue size for input channels";

/// @brief message for FPS measurement sampling period
static const char fps_sampling_period[] = "FPS measurement sampling period. Duration between timepoints, msec";

/// @brief message for FPS measurement sampling period
static const char num_sampling_periods[] = "Number of sampling periods";

/// @brief message for enabling statistics output
static const char show_statistics[] = "Enable statictics output";

/// @brief message for enabling channel duplication
static const char duplication_channel_number[] = "Enable and specify number of channel additionally copied from real sources";

/// @brief message for enabling real input FPS
static const char real_input_fps[] = "Disable input frames caching, for maximum throughput pipeline";

/// @brief message for enabling input video
static const char input_video[] = "Specify full path to input video files";

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

/// \brief Flag to output raw scoring results<br>
/// It is an optional parameter
DEFINE_double(t, 0.5, thresh_output_message);

/// \brief Flag to disable processed video showing<br>
/// It is an optional parameter
DEFINE_bool(no_show, false, no_show_processed_video);

/// \brief Flag to specify number of expected input channels<br>
/// It is an optional parameter
DEFINE_uint32(nc, 4, num_cameras);

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
