#pragma once
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

#include <gflags/gflags.h>

/// @brief message for help argument
static const char help_message[] = "Print a usage message";
/// @brief message for images argument
static const char image_message[] = "Required. Path to input video file";
/// @brief message for model argument
static const char model_message[] = "Required. Path to IR .xml file.";
/// @brief message for labels argument
static const char labels_message[] = "Required. Path to labels file.";
/// @brief message for threshold
static const char threshold_message[] = "Confidence threshold for bounding boxes 0-1";
/// @brief message for frames count
static const char frames_message[] = "Number of frames from stream to process";
/// @brief message for verbose
static const char useclasses_message[] = "Flag: 1 if class is used, 0 if not";
/// @brief message for output width
static const char output_width_message[] = "Width of output frames";
/// @brief message for output height
static const char output_height_message[] = "Height of output frames";
/// @brief message for plugin_path argument
static const char plugin_path_message[] = "Path to a plugin folder";
/// @brief message for batch size
static const char batch_message[] = "Batch size";
/// @brief message for plugin argument
static const char plugin_message[] = "Plugin name. (MKLDNNPlugin, clDNNPlugin) Force load specified plugin ";
/// @brief message for assigning cnn calculation to device
static const char target_device_message[] = "Infer target device (CPU or GPU or MYRIAD)";
/// @brief message for performance counters
static const char performance_counter_message[] = "Enables per-layer performance report";
/// @brief message for silence
static const char simple_output_message[] = "Display less information on the screen";
/// @brief message for loading layer extension plugin
static const char layer_extension_plugin_message[] = "Load layer extension plugin";
/// @brief message for mean value used for normalization
static const char mean_message[] = "Mean value for normalization of data during planar BGR blob preprocess step";
/// @brief message for scale value used for normalization
static const char scale_message[] = "Scale value for normalization of data during planar BGR blob preprocess step";

/// \brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

#ifdef WIN32
    /// \brief Define parameter for video file <br>
    /// It is a required parameter
    DEFINE_string(i, "../../../samples/end2end_video_analytics/test_content/video/cars_1920x1080.h264", image_message);
    /// \brief Define parameter for set model file <br>
    /// It is a required parameter
    DEFINE_string(m, "", model_message);
    /// \brief Define parameter for labels file <br>
    /// It is a required parameter
    DEFINE_string(l, "../../../samples/end2end_video_analytics/test_content/IR/SSD/pascal_voc_classes.txt", labels_message);
#else
    /// \brief Define parameter for video file <br>
    /// It is a required parameter
    DEFINE_string(i, "../../../end2end_video_analytics/test_content/video/cars_1920x1080.h264", image_message);
    /// \brief Define parameter for set model file <br>
    /// It is a required parameter
    DEFINE_string(m, "", model_message);
    /// \brief Define parameter for labels file <br>
    /// It is a required parameter
    DEFINE_string(l, "../../../end2end_video_analytics/test_content/IR/SSD/pascal_voc_classes.txt", labels_message);
#endif

/// \brief Define parameter for set plugin name <br>
/// It is a required parameter
DEFINE_string(p, "", plugin_message);
/// \brief device the target device to infer on <br>
DEFINE_string(d, "CPU", target_device_message);
/// \brief Enable per-layer performance report
DEFINE_bool(pc, false, performance_counter_message);
/// \brief threshold for bounding boxes
DEFINE_double(thresh, .4, threshold_message);
/// \brief Batch size
DEFINE_int32(batch, 1, batch_message);
/// \brief Frames count
DEFINE_int32(fr, 256, frames_message);
/// \brief use/not use flag for classes
DEFINE_string(useclasses, "00000010000000100000", useclasses_message);
/// \brief output width
DEFINE_int32(output_w, 300, output_width_message);
/// \brief output height
DEFINE_int32(output_h, 300, output_height_message);
/// \brief simple output <br>
DEFINE_bool(s, false, simple_output_message);
/// \brief layer extension plugin <br>
DEFINE_string(e, "", layer_extension_plugin_message);
/// \brief the Mean to use for normalization
DEFINE_double(mean, 0, mean_message);
/// \brief  the Scale to use for normalization
DEFINE_double(scale, 1, scale_message);


