// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "core.hpp"
#include "logging.hpp"

#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>
#include <deque>
#include <map>

#include <samples/common.hpp>

///
/// \brief The DetectionLogEntry struct
///
/// An entry describing detected objects on a frame.
///
struct DetectionLogEntry {
    TrackedObjects objects;  ///< Detected objects.
    int frame_idx;           ///< Processed frame index (-1 if N/A).
    double time_ms;          ///< Frame processing time in ms (-1 if N/A).

    ///
    /// \brief DetectionLogEntry default constructor.
    ///
    DetectionLogEntry() : frame_idx(-1), time_ms(-1) {}

    ///
    /// \brief DetectionLogEntry copy constructor.
    /// \param other Detection entry.
    ///
    DetectionLogEntry(const DetectionLogEntry &other)
        : objects(other.objects),
        frame_idx(other.frame_idx),
        time_ms(other.time_ms) {}

    ///
    /// \brief DetectionLogEntry move constructor.
    /// \param other Detection entry.
    ///
    DetectionLogEntry(DetectionLogEntry &&other)
        : objects(std::move(other.objects)),
        frame_idx(other.frame_idx),
        time_ms(other.time_ms) {}

    ///
    /// \brief Assignment operator.
    /// \param other Detection entry.
    /// \return Detection entry.
    ///
    DetectionLogEntry &operator=(const DetectionLogEntry &other) = default;

    ///
    /// \brief Move assignment operator.
    /// \param other Detection entry.
    /// \return Detection entry.
    ///
    DetectionLogEntry &operator=(DetectionLogEntry &&other) {
        if (this != &other) {
            objects = std::move(other.objects);
            frame_idx = other.frame_idx;
            time_ms = other.time_ms;
        }
        return *this;
    }
};

/// Detection log is a vector of detection entries.
using DetectionLog = std::vector<DetectionLogEntry>;

///
/// \brief Save DetectionLog to a txt file in the format
///        compatible with the format of MOTChallenge
///        evaluation tool.
/// \param[in] path -- path to a file to store
/// \param[in] log  -- detection log to store
///
void SaveDetectionLogToTrajFile(const std::string& path,
                                const DetectionLog& log);

///
/// \brief Print DetectionLog to stdout in the format
///        compatible with the format of MOTChallenge
///        evaluation tool.
/// \param[in] log  -- detection log to print
///
void PrintDetectionLog(const DetectionLog& log);

///
/// \brief Draws a polyline on a frame.
/// \param[in] polyline Vector of points (polyline).
/// \param[in] color Color (BGR).
/// \param[in,out] image Frame.
/// \param[in] lwd Line width.
///
void DrawPolyline(const std::vector<cv::Point>& polyline,
                  const cv::Scalar& color, cv::Mat* image,
                  int lwd = 5);

///
/// \brief Stream output operator for deque of elements.
/// \param[in,out] os Output stream.
/// \param[in] v Vector of elements.
///
template <typename T>
std::ostream &operator<<(std::ostream &os, const std::deque<T> &v) {
    os << "[\n";
    if (!v.empty()) {
        auto itr = v.begin();
        os << *itr;
        for (++itr; itr != v.end(); ++itr) os << ",\n" << *itr;
    }
    os << "\n]";
    return os;
}

///
/// \brief Stream output operator for vector of elements.
/// \param[in,out] os Output stream.
/// \param[in] v Vector of elements.
///
template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
    os << "[\n";
    if (!v.empty()) {
        auto itr = v.begin();
        os << *itr;
        for (++itr; itr != v.end(); ++itr) os << ",\n" << *itr;
    }
    os << "\n]";
    return os;
}

InferenceEngine::Core
LoadInferenceEngine(const std::vector<std::string>& devices,
                    const std::string& custom_cpu_library,
                    const std::string& custom_cldnn_kernels,
                    bool should_use_perf_counter);
