// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <deque>
#include <iostream>
#include <string>
#include <unordered_map>

#include <opencv2/core.hpp>

///
/// \brief The TrackedObject struct defines properties of detected object.
///
struct TrackedObject {
    cv::Rect rect;  ///< Detected object ROI (zero area if N/A).
    double confidence;  ///< Detection confidence level (-1 if N/A).
    int64_t frame_idx;  ///< Frame index where object was detected (-1 if N/A).
    int object_id;  ///< Unique object identifier (-1 if N/A).
    uint64_t timestamp;  ///< Timestamp in milliseconds.

    ///
    /// \brief Default constructor.
    ///
    TrackedObject() : confidence(-1), frame_idx(-1), object_id(-1), timestamp(0) {}

    ///
    /// \brief Constructor with parameters.
    /// \param rect Bounding box of detected object.
    /// \param confidence Confidence of detection.
    /// \param frame_idx Index of frame.
    /// \param object_id Object ID.
    ///
    TrackedObject(const cv::Rect& rect, float confidence, int64_t frame_idx, int object_id)
        : rect(rect),
          confidence(confidence),
          frame_idx(frame_idx),
          object_id(object_id),
          timestamp(0) {}
};

using TrackedObjects = std::deque<TrackedObject>;

/// (object id, detected objects) pairs collection.
using ObjectTracks = std::unordered_map<int, TrackedObjects>;
