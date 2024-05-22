// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>

#include <set>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>

struct TrackedObject {
    cv::Rect rect;
    float confidence = -1.0f;
    int object_id = -1;
    size_t frame_idx = -1;
};

using TrackedObjects = std::vector<TrackedObject>;

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

    bool drop_forgotten_tracks;  ///< Drop forgotten tracks. If it's enabled it
    /// disables an ability to get detection log.

    int max_num_objects_in_track;  ///< The number of objects in track is
    /// restricted by this parameter. If it is negative or zero, the max number of
    /// objects in track is not restricted.

    int averaging_window_size_for_rects;  ///< The number of objects in track for averaging rects of predictions.
    int averaging_window_size_for_labels;  ///< The number of objects in track for averaging labels of predictions.

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
    explicit Track(const TrackedObjects& objs) : objects(objs), lost(0), length(1) {
        CV_Assert(!objs.empty());
        first_object = objs[0];
    }

    ///
    /// \brief empty returns if track does not contain objects.
    /// \return true if track does not contain objects.
    ///
    bool empty() const {
        return objects.empty();
    }

    ///
    /// \brief size returns number of detected objects in a track.
    /// \return number of detected objects in a track.
    ///
    size_t size() const {
        return objects.size();
    }

    ///
    /// \brief operator [] return const reference to detected object with
    ///        specified index.
    /// \param i Index of object.
    /// \return const reference to detected object with specified index.
    ///
    const TrackedObject& operator[](size_t i) const {
        return objects[i];
    }

    ///
    /// \brief operator [] return non-const reference to detected object with
    ///        specified index.
    /// \param i Index of object.
    /// \return non-const reference to detected object with specified index.
    ///
    TrackedObject& operator[](size_t i) {
        return objects[i];
    }

    ///
    /// \brief back returns const reference to last object in track.
    /// \return const reference to last object in track.
    ///
    const TrackedObject& back() const {
        CV_Assert(!empty());
        return objects.back();
    }

    ///
    /// \brief back returns non-const reference to last object in track.
    /// \return non-const reference to last object in track.
    ///
    TrackedObject& back() {
        CV_Assert(!empty());
        return objects.back();
    }

    TrackedObjects objects;  ///< Detected objects;
    size_t lost;  ///< How many frames ago track has been lost.

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
    explicit Tracker(const TrackerParams& params = TrackerParams())
        : params_(params),
          tracks_counter_(0),
          frame_size_() {}

    ///
    /// \brief process given frame.
    /// \param[in] frame Colored image (CV_8UC3).
    /// \param[in] detections Detected objects on the frame.
    /// \param[in] timestamp Timestamp must be positive and measured in
    /// milliseconds
    ///
    void process(const cv::Mat& frame, const TrackedObjects& detections);

    ///
    /// \brief Get tracked detections with labels.
    /// \return Tracked detections.
    ///
    TrackedObjects trackedDetectionsWithLabels() const;

    ///
    /// \brief isTrackForgotten returns true if track is forgotten.
    /// \param id Track ID.
    /// \return true if track is forgotten.
    ///
    bool isTrackForgotten(size_t id) const;

    ///
    /// \brief tracks Returns all tracks including forgotten (lost too many frames
    /// ago).
    /// \return Set of tracks {id, track}.
    ///
    const std::unordered_map<size_t, Track>& tracks() const;

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
    bool isTrackValid(size_t id) const;

    ///
    /// \brief DropForgottenTracks Removes tracks from memory that were lost too
    /// many frames ago.
    ///
    void dropForgottenTracks();

private:
    const std::set<size_t>& active_track_ids() const {
        return active_track_ids_;
    }

    float shapeAffinity(const cv::Rect& trk, const cv::Rect& det);
    float motionAffinity(const cv::Rect& trk, const cv::Rect& det);

    void solveAssignmentProblem(const std::set<size_t>& track_ids,
                                const TrackedObjects& detections,
                                std::set<size_t>* unmatched_tracks,
                                std::set<size_t>* unmatched_detections,
                                std::set<std::tuple<size_t, size_t, float>>* matches);

    void computeDissimilarityMatrix(const std::set<size_t>& active_track_ids,
                                    const TrackedObjects& detections,
                                    cv::Mat* dissimilarity_matrix);

    float distance(const TrackedObject& obj1, const TrackedObject& obj2);

    void addNewTrack(const TrackedObject& detection);

    void addNewTracks(const TrackedObjects& detections);

    void addNewTracks(const TrackedObjects& detections, const std::set<size_t>& ids);

    void appendToTrack(size_t track_id, const TrackedObject& detection);

    bool eraseTrackIfBBoxIsOutOfFrame(size_t track_id);

    bool eraseTrackIfItWasLostTooManyFramesAgo(size_t track_id);

    bool uptateLostTrackAndEraseIfItsNeeded(size_t track_id);

    void updateLostTracks(const std::set<size_t>& track_ids);

    // Parameters of the pipeline.
    TrackerParams params_;

    // Indexes of active tracks.
    std::set<size_t> active_track_ids_;

    // All tracks.
    std::unordered_map<size_t, Track> tracks_;

    // Number of all current tracks.
    size_t tracks_counter_;

    cv::Size frame_size_;

    size_t frame_idx_ = 0;
};
