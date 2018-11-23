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
static const char model_message[] = "Required. Path to Caffe deploy.prototxt file.";
/// @brief message for weights argument
static const char weights_message[] = "Required. Path to Caffe weights in .caffemodel file.";
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
/// @brief message for slience
static const char simple_output_message[] = "Display less information on the screen";

/// \brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

#ifdef WIN32
    /// \brief Define parameter for video file <br>
    /// It is a required parameter
    DEFINE_string(i, "../../../samples/end2end_video_analytics/test_content/video/cars_1920x1080.h264", image_message);
    /// \brief Define parameter for model file <br>
    /// It is a required parameter
    DEFINE_string(m, "", model_message);
    /// \brief Define parameter for weights file (.caffemodel) <br>
    /// It is a required parameter
    DEFINE_string(weights, "", weights_message);
    /// \brief Define parameter for labels file <br>
    /// It is a required parameter
    DEFINE_string(l, "../../../samples/end2end_video_analytics/test_content/IR/SSD/pascal_voc_classes.txt", labels_message);
#else
    /// \brief Define parameter for video file <br>
    /// It is a required parameter
    DEFINE_string(i, "../../../end2end_video_analytics/test_content/video/cars_1920x1080.h264", image_message);
    /// \brief Define parameter for model file <br>
    /// It is a required parameter
    DEFINE_string(m, "", model_message);
    /// \brief Define parameter for weights file (.caffemodel) <br>
    /// It is a required parameter
    DEFINE_string(weights, "", weights_message);
    /// \brief Define parameter for labels file <br>
    /// It is a required parameter
    DEFINE_string(l, "../../../end2end_video_analytics/test_content/IR/SSD/pascal_voc_classes.txt", labels_message);
#endif

/// \brief threshold for bounding boxes
DEFINE_double(thresh, .4, threshold_message);
/// \brief Frames count
DEFINE_int32(fr, 256, frames_message);
/// \brief use/not use flag for classes
DEFINE_string(useclasses, "00000010000000100000", useclasses_message);
/// \brief output width
DEFINE_int32(output_w, 300, output_width_message);
/// \brief output height
DEFINE_int32(output_h, 300, output_height_message);
// temporary workaround for h264 encode outliers
DEFINE_double(max_encode_ms, 20.0, NULL);
/// \brief simple output <br>
DEFINE_bool(s, false, simple_output_message);

