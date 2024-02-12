// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tracker.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <set>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <utils/kuhn_munkres.hpp>

cv::Point center(const cv::Rect& rect) {
    int x = static_cast<int>(rect.x + rect.width * 0.5);
    int y = static_cast<int>(rect.y + rect.height * 0.5);
    CV_DbgAssert(x);
    CV_DbgAssert(y);
    return cv::Point(x, y);
}

// FIXME: Need to simplify tracker for this demo
TrackerParams::TrackerParams()
    : min_track_duration(25),
      forget_delay(150),
      affinity_thr(0.85f),
      shape_affinity_w(0.5f),
      motion_affinity_w(0.2f),
      drop_forgotten_tracks(true),
      max_num_objects_in_track(300),
      averaging_window_size_for_rects(1),
      averaging_window_size_for_labels(1) {}

void Tracker::solveAssignmentProblem(const std::set<size_t>& track_ids,
                                     const TrackedObjects& detections,
                                     std::set<size_t>* unmatched_tracks,
                                     std::set<size_t>* unmatched_detections,
                                     std::set<std::tuple<size_t, size_t, float>>* matches) {
    CV_Assert(unmatched_tracks);
    CV_Assert(unmatched_detections);
    unmatched_tracks->clear();
    unmatched_detections->clear();

    CV_Assert(!track_ids.empty());
    CV_Assert(!detections.empty());
    CV_Assert(matches);
    matches->clear();

    cv::Mat dissimilarity;
    computeDissimilarityMatrix(track_ids, detections, &dissimilarity);

    auto res = KuhnMunkres().Solve(dissimilarity);

    for (size_t i = 0; i < detections.size(); i++) {
        unmatched_detections->insert(i);
    }

    size_t i = 0;
    for (size_t id : track_ids) {
        if (res[i] < detections.size()) {
            matches->emplace(id, res[i], 1 - dissimilarity.at<float>(i, res[i]));
        } else {
            unmatched_tracks->insert(id);
        }
        i++;
    }
}

bool Tracker::eraseTrackIfBBoxIsOutOfFrame(size_t track_id) {
    if (tracks_.find(track_id) == tracks_.end())
        return true;
    auto c = center(tracks_.at(track_id).back().rect);
    if (frame_size_ != cv::Size() && (c.x < 0 || c.y < 0 || c.x > frame_size_.width || c.y > frame_size_.height)) {
        tracks_.at(track_id).lost = params_.forget_delay + 1;
        active_track_ids_.erase(track_id);
        return true;
    }
    return false;
}

bool Tracker::eraseTrackIfItWasLostTooManyFramesAgo(size_t track_id) {
    if (tracks_.find(track_id) == tracks_.end())
        return true;
    if (tracks_.at(track_id).lost > params_.forget_delay) {
        active_track_ids_.erase(track_id);
        return true;
    }
    return false;
}

bool Tracker::uptateLostTrackAndEraseIfItsNeeded(size_t track_id) {
    tracks_.at(track_id).lost++;
    bool erased = eraseTrackIfBBoxIsOutOfFrame(track_id);
    if (!erased)
        erased = eraseTrackIfItWasLostTooManyFramesAgo(track_id);
    return erased;
}

void Tracker::updateLostTracks(const std::set<size_t>& track_ids) {
    for (auto track_id : track_ids) {
        uptateLostTrackAndEraseIfItsNeeded(track_id);
    }
}

void Tracker::process(const cv::Mat& frame, const TrackedObjects& detections) {
    if (frame_size_ == cv::Size()) {
        frame_size_ = frame.size();
    } else {
        CV_Assert(frame_size_ == frame.size());
    }

    for (auto& obj : detections) {
        *const_cast<size_t*>(&obj.frame_idx) = frame_idx_;
    }

    ++frame_idx_;
    auto active_tracks = active_track_ids_;

    if (!active_tracks.empty() && !detections.empty()) {
        std::set<size_t> unmatched_tracks, unmatched_detections;
        std::set<std::tuple<size_t, size_t, float>> matches;

        solveAssignmentProblem(active_tracks, detections, &unmatched_tracks, &unmatched_detections, &matches);

        for (const auto& match : matches) {
            size_t track_id = std::get<0>(match);
            size_t det_id = std::get<1>(match);
            float conf = std::get<2>(match);
            if (conf > params_.affinity_thr) {
                appendToTrack(track_id, detections[det_id]);
                unmatched_detections.erase(det_id);
            } else {
                unmatched_tracks.insert(track_id);
            }
        }

        addNewTracks(detections, unmatched_detections);
        updateLostTracks(unmatched_tracks);

        for (size_t id : active_tracks) {
            eraseTrackIfBBoxIsOutOfFrame(id);
        }
    } else {
        addNewTracks(detections);
        updateLostTracks(active_tracks);
    }

    if (params_.drop_forgotten_tracks)
        dropForgottenTracks();
}

void Tracker::dropForgottenTracks() {
    std::unordered_map<size_t, Track> new_tracks;
    std::set<size_t> new_active_tracks;

    size_t max_id = 0;
    if (!active_track_ids_.empty())
        max_id = *std::max_element(active_track_ids_.begin(), active_track_ids_.end());

    const size_t kMaxTrackID = 10000;
    bool reassign_id = max_id > kMaxTrackID;

    size_t counter = 0;
    for (const auto& pair : tracks_) {
        if (!isTrackForgotten(pair.first)) {
            new_tracks.emplace(reassign_id ? counter : pair.first, pair.second);
            new_active_tracks.emplace(reassign_id ? counter : pair.first);
            counter++;
        }
    }
    tracks_.swap(new_tracks);
    active_track_ids_.swap(new_active_tracks);

    tracks_counter_ = reassign_id ? counter : tracks_counter_;
}

float Tracker::shapeAffinity(const cv::Rect& trk, const cv::Rect& det) {
    float w_dist = static_cast<float>(std::fabs(trk.width - det.width)) / static_cast<float>(trk.width + det.width);
    float h_dist = static_cast<float>(std::fabs(trk.height - det.height)) / static_cast<float>(trk.height + det.height);
    return exp(-params_.shape_affinity_w * (w_dist + h_dist));
}

float Tracker::motionAffinity(const cv::Rect& trk, const cv::Rect& det) {
    float x_dist = static_cast<float>(trk.x - det.x) * (trk.x - det.x) / (det.width * det.width);
    float y_dist = static_cast<float>(trk.y - det.y) * (trk.y - det.y) / (det.height * det.height);
    return exp(-params_.motion_affinity_w * (x_dist + y_dist));
}

void Tracker::computeDissimilarityMatrix(const std::set<size_t>& active_tracks,
                                         const TrackedObjects& detections,
                                         cv::Mat* dissimilarity_matrix) {
    dissimilarity_matrix->create(active_tracks.size(), detections.size(), CV_32F);
    size_t i = 0;
    for (auto id : active_tracks) {
        auto ptr = dissimilarity_matrix->ptr<float>(i);
        for (size_t j = 0; j < detections.size(); j++) {
            auto last_det = tracks_.at(id).objects.back();
            ptr[j] = distance(last_det, detections[j]);
        }
        i++;
    }
}

void Tracker::addNewTracks(const TrackedObjects& detections) {
    for (size_t i = 0; i < detections.size(); i++) {
        addNewTrack(detections[i]);
    }
}

void Tracker::addNewTracks(const TrackedObjects& detections, const std::set<size_t>& ids) {
    for (size_t i : ids) {
        CV_Assert(i < detections.size());
        addNewTrack(detections[i]);
    }
}

void Tracker::addNewTrack(const TrackedObject& detection) {
    auto detection_with_id = detection;
    detection_with_id.object_id = tracks_counter_;
    tracks_.emplace(std::pair<size_t, Track>(tracks_counter_, Track({detection_with_id})));

    active_track_ids_.insert(tracks_counter_);
    tracks_counter_++;
}

void Tracker::appendToTrack(size_t track_id, const TrackedObject& detection) {
    CV_Assert(!isTrackForgotten(track_id));

    auto detection_with_id = detection;
    detection_with_id.object_id = track_id;

    auto& track = tracks_.at(track_id);

    track.objects.emplace_back(detection_with_id);
    track.lost = 0;
    track.length++;

    if (params_.max_num_objects_in_track > 0) {
        while (track.size() > static_cast<size_t>(params_.max_num_objects_in_track)) {
            track.objects.erase(track.objects.begin());
        }
    }
}

float Tracker::distance(const TrackedObject& obj1, const TrackedObject& obj2) {
    const float eps = 1e-6f;
    float shp_aff = shapeAffinity(obj1.rect, obj2.rect);
    if (shp_aff < eps)
        return 1.0;

    float mot_aff = motionAffinity(obj1.rect, obj2.rect);
    if (mot_aff < eps)
        return 1.0;

    return 1.0f - shp_aff * mot_aff;
}

bool Tracker::isTrackValid(size_t id) const {
    const auto& track = tracks_.at(id);
    const auto& objects = track.objects;
    if (objects.empty()) {
        return false;
    }
    size_t duration_frames = objects.back().frame_idx - track.first_object.frame_idx;
    if (duration_frames < params_.min_track_duration)
        return false;
    return true;
}

bool Tracker::isTrackForgotten(size_t id) const {
    return tracks_.at(id).lost > params_.forget_delay;
}

TrackedObjects Tracker::trackedDetectionsWithLabels() const {
    TrackedObjects detections;

    for (size_t idx : active_track_ids()) {
        const auto& track = tracks().at(idx);
        if (isTrackValid(idx) && !track.lost) {
            TrackedObject object = track.objects.back();
            int counter = 1;
            size_t start = static_cast<int>(track.objects.size()) >= params_.averaging_window_size_for_rects
                               ? track.objects.size() - params_.averaging_window_size_for_rects
                               : 0;

            for (size_t i = start; i < track.objects.size() - 1; i++) {
                object.rect.width += track.objects[i].rect.width;
                object.rect.height += track.objects[i].rect.height;
                object.rect.x += track.objects[i].rect.x;
                object.rect.y += track.objects[i].rect.y;
                counter++;
            }
            object.rect.width /= counter;
            object.rect.height /= counter;
            object.rect.x /= counter;
            object.rect.y /= counter;

            object.object_id = idx;

            detections.push_back(object);
        }
    }
    return detections;
}

const std::unordered_map<size_t, Track>& Tracker::tracks() const {
    return tracks_;
}
