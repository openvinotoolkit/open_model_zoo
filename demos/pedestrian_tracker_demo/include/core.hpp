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

#include <opencv2/core.hpp>

#include <deque>
#include <iostream>
#include <string>
#include <unordered_map>

///
/// \brief The TrackedObject struct defines properties of detected object.
///
struct TrackedObject {
    cv::Rect rect;       ///< Detected object ROI (zero area if N/A).
    double confidence;   ///< Detection confidence level (-1 if N/A).
    int frame_idx;       ///< Frame index where object was detected (-1 if N/A).
    int object_id;       ///< Unique object identifier (-1 if N/A).
    uint64_t timestamp;  ///< Timestamp in milliseconds.

    ///
    /// \brief Default constructor.
    ///
    TrackedObject()
        : confidence(-1),
        frame_idx(-1),
        object_id(-1),
        timestamp(0) {}

    ///
    /// \brief Constructor with parameters.
    /// \param rect Bounding box of detected object.
    /// \param confidence Confidence of detection.
    /// \param frame_idx Index of frame.
    /// \param object_id Object ID.
    ///
    TrackedObject(const cv::Rect &rect, float confidence, int frame_idx,
                  int object_id)
        : rect(rect),
        confidence(confidence),
        frame_idx(frame_idx),
        object_id(object_id),
        timestamp(0) {}
};

using TrackedObjects = std::deque<TrackedObject>;

bool operator==(const TrackedObject& first, const TrackedObject& second);
bool operator!=(const TrackedObject& first, const TrackedObject& second);
/// (object id, detected objects) pairs collection.
using ObjectTracks = std::unordered_map<int, TrackedObjects>;

