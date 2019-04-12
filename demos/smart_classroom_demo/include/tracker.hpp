// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cnn.hpp"

#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

struct TrackedObject {
    cv::Rect rect;
    float confidence;

    int object_id;
    int label;  // either id of a label, or UNKNOWN_LABEL_IDX
    static const int UNKNOWN_LABEL_IDX;  // the value (-1) for unknown label

    size_t frame_idx;      ///< Frame index where object was detected (-1 if N/A).

    TrackedObject(const cv::Rect &rect = cv::Rect(), float conf = -1.0f,
                  int label = -1, int object_id = -1)
        : rect(rect),  confidence(conf),
          object_id(object_id), label(label),
          frame_idx(-1) {}
};

using TrackedObjects = std::vector<TrackedObject>;

///
/// \brief The KuhnMunkres class
///
/// Solves the assignment problem.
///
class KuhnMunkres {
public:
    KuhnMunkres();

    ///
    /// \brief Solves the assignment problem for given dissimilarity matrix.
    /// It returns a vector that where each element is a column index for
    /// corresponding row (e.g. result[0] stores optimal column index for very
    /// first row in the dissimilarity matrix).
    /// \param dissimilarity_matrix CV_32F dissimilarity matrix.
    /// \return Optimal column index for each row. -1 means that there is no
    /// column for row.
    ///
    std::vector<size_t> Solve(const cv::Mat &dissimilarity_matrix);

private:
    class Impl;
    std::shared_ptr<Impl> impl_;  ///< Class implementation.
};

///
/// \brief The Params struct stores parameters of Tracker.
///
struct TrackerParams {
    size_t min_track_duration;  ///< Min track duration in frames

    size_t forget_delay;  ///< Forget about track if the last bounding box in
    /// track was detected more than specified number of
    /// frames ago.

    float affinity_thr;  ///< Affinity threshold which is used to determine if
    /// tracklet and detection should be combined.

    float shape_affinity_w;  ///< Shape affinity weight.

    float motion_affinity_w;  ///< Motion affinity weight.

    float min_det_conf;  ///< Min confidence of detection.

    cv::Vec2f bbox_aspect_ratios_range;  ///< Bounding box aspect ratios range.

    cv::Vec2f bbox_heights_range;  ///< Bounding box heights range.

    bool drop_forgotten_tracks;  ///< Drop forgotten tracks. If it's enabled it
    /// disables an ability to get detection log.

    int max_num_objects_in_track;  ///< The number of objects in track is
    /// restricted by this parameter. If it is negative or zero, the max number of
    /// objects in track is not restricted.

    int averaging_window_size_for_rects;  ///< The number of objects in track for averaging rects of predictions.
    int averaging_window_size_for_labels;  ///< The number of objects in track for averaging labels of predictions.

    std::string objects_type;  ///< The type of boxes which will be grabbed from
    /// detector. Boxes with other types are ignored.

    ///
    /// Default constructor.
    ///
    TrackerParams();
};

///
/// \brief The Track struct describes tracks.
///
struct Track {
    ///
    /// \brief Track constructor.
    /// \param objs Detected objects sequence.
    ///
    explicit Track(const TrackedObjects &objs) : objects(objs), lost(0), length(1) {
        CV_Assert(!objs.empty());
        first_object = objs[0];
    }

    ///
    /// \brief empty returns if track does not contain objects.
    /// \return true if track does not contain objects.
    ///
    bool empty() const { return objects.empty(); }

    ///
    /// \brief size returns number of detected objects in a track.
    /// \return number of detected objects in a track.
    ///
    size_t size() const { return objects.size(); }

    ///
    /// \brief operator [] return const reference to detected object with
    ///        specified index.
    /// \param i Index of object.
    /// \return const reference to detected object with specified index.
    ///
    const TrackedObject &operator[](size_t i) const { return objects[i]; }

    ///
    /// \brief operator [] return non-const reference to detected object with
    ///        specified index.
    /// \param i Index of object.
    /// \return non-const reference to detected object with specified index.
    ///
    TrackedObject &operator[](size_t i) { return objects[i]; }

    ///
    /// \brief back returns const reference to last object in track.
    /// \return const reference to last object in track.
    ///
    const TrackedObject &back() const {
        CV_Assert(!empty());
        return objects.back();
    }

    ///
    /// \brief back returns non-const reference to last object in track.
    /// \return non-const reference to last object in track.
    ///
    TrackedObject &back() {
        CV_Assert(!empty());
        return objects.back();
    }

    TrackedObjects objects;  ///< Detected objects;
    size_t lost;             ///< How many frames ago track has been lost.

    TrackedObject first_object;  ///< First object in track.
    size_t length;  ///< Length of a track including number of objects that were
                    /// removed from track in order to avoid memory usage growth.
};

///
/// \brief Simple Hungarian algorithm-based tracker.
///
class Tracker {
public:
    ///
    /// \brief Constructor that creates an instance of Tracker with
    /// parameters.
    /// \param[in] params Tracker parameters.
    ///
    explicit Tracker(const TrackerParams &params = TrackerParams())
        : params_(params),
          tracks_counter_(0),
          valid_tracks_counter_(0),
          frame_size_() {}

    ///
    /// \brief Process given frame.
    /// \param[in] frame Colored image (CV_8UC3).
    /// \param[in] detections Detected objects on the frame.
    /// \param[in] timestamp Timestamp must be positive and measured in
    /// milliseconds
    ///
    void Process(const cv::Mat &frame, const TrackedObjects &detections,
                 int frame_idx);

    ///
    /// \brief Pipeline parameters getter.
    /// \return Parameters of pipeline.
    ///
    const TrackerParams &params() const;

    ///
    /// \brief Pipeline parameters setter.
    /// \param[in] params Parameters of pipeline.
    ///
    void set_params(const TrackerParams &params);

    ///
    /// \brief Reset the pipeline.
    ///
    void Reset();

    ///
    /// \brief Returns number of counted tracks.
    /// \return a number of counted tracks.
    ///
    size_t Count() const;

    ///
    /// \brief Returns recently detected objects.
    /// \return recently detected objects.
    ///
    const TrackedObjects &detections() const;

    ///
    /// \brief Get active tracks to draw
    /// \return Active tracks.
    ///
    std::unordered_map<size_t, std::vector<cv::Point>> GetActiveTracks() const;

    ///
    /// \brief Get tracked detections.
    /// \return Tracked detections.
    ///
    TrackedObjects TrackedDetections() const;

    ///
    /// \brief Get tracked detections with labels.
    /// \return Tracked detections.
    ///
    TrackedObjects TrackedDetectionsWithLabels() const;

    ///
    /// \brief IsTrackForgotten returns true if track is forgotten.
    /// \param id Track ID.
    /// \return true if track is forgotten.
    ///
    bool IsTrackForgotten(size_t id) const;

    ///
    /// \brief tracks Returns all tracks including forgotten (lost too many frames
    /// ago).
    /// \return Set of tracks {id, track}.
    ///
    const std::unordered_map<size_t, Track> &tracks() const;

    ///
    /// \brief tracks Returns all tracks including forgotten (lost too many frames
    /// ago).
    /// \return Vector of tracks
    ///
    std::vector<Track> vector_tracks() const;

    ///
    /// \brief IsTrackValid Checks whether track is valid (duration > threshold).
    /// \param id Index of checked track.
    /// \return True if track duration exceeds some predefined value.
    ///
    bool IsTrackValid(size_t id) const;

    ///
    /// \brief DropForgottenTracks Removes tracks from memory that were lost too
    /// many frames ago.
    ///
    void DropForgottenTracks();

private:
    void DropForgottenTrack(size_t track_id);

    const std::set<size_t> &active_track_ids() const { return active_track_ids_; }

    float ShapeAffinity(const cv::Rect &trk, const cv::Rect &det);
    float MotionAffinity(const cv::Rect &trk, const cv::Rect &det);

    void SolveAssignmentProblem(
            const std::set<size_t> &track_ids, const TrackedObjects &detections,
            std::set<size_t> *unmatched_tracks,
            std::set<size_t> *unmatched_detections,
            std::set<std::tuple<size_t, size_t, float>> *matches);
    void FilterDetectionsAndStore(const TrackedObjects &detected_objects);

    void ComputeDissimilarityMatrix(const std::set<size_t> &active_track_ids,
                                    const TrackedObjects &detections,
                                    cv::Mat *dissimilarity_matrix);

    std::vector<std::pair<size_t, size_t>> GetTrackToDetectionIds(
            const std::set<std::tuple<size_t, size_t, float>> &matches);

    float Distance(const TrackedObject &obj1, const TrackedObject &obj2);

    void AddNewTrack(const TrackedObject &detection);

    void AddNewTracks(const TrackedObjects &detections);

    void AddNewTracks(const TrackedObjects &detections,
                      const std::set<size_t> &ids);

    void AppendToTrack(size_t track_id, const TrackedObject &detection);

    bool EraseTrackIfBBoxIsOutOfFrame(size_t track_id);

    bool EraseTrackIfItWasLostTooManyFramesAgo(size_t track_id);

    bool UptateLostTrackAndEraseIfItsNeeded(size_t track_id);

    void UpdateLostTracks(const std::set<size_t> &track_ids);

    std::unordered_map<size_t, std::vector<cv::Point>> GetActiveTracks();

    // Parameters of the pipeline.
    TrackerParams params_;

    // Indexes of active tracks.
    std::set<size_t> active_track_ids_;

    // All tracks.
    std::unordered_map<size_t, Track> tracks_;

    // Recent detections.
    TrackedObjects detections_;

    // Number of all current tracks.
    size_t tracks_counter_;

    // Number of dropped valid tracks.
    size_t valid_tracks_counter_;

    cv::Size frame_size_;
};

int LabelWithMaxFrequencyInTrack(const Track &track, int window_size);
std::vector<Track> UpdateTrackLabelsToBestAndFilterOutUnknowns(const std::vector<Track>& tracks);
