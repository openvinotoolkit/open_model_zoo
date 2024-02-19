// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>
#include <tuple>
#include <set>
#include <utility>
#include <unordered_map>

#include "logger.hpp"
#include "tracker.hpp"

const int TrackedObject::UNKNOWN_LABEL_IDX = -1;

cv::Point Center(const cv::Rect& rect) {
    return cv::Point(static_cast<int>(rect.x + rect.width * 0.5),
                     static_cast<int>(rect.y + rect.height * 0.5));
}

TrackerParams::TrackerParams() :
    min_track_duration(25),
    forget_delay(150),
    affinity_thr(0.85f),
    shape_affinity_w(0.5f),
    motion_affinity_w(0.2f),
    min_det_conf(0.0f),
    bbox_aspect_ratios_range(0.666f, 5.0f),
    bbox_heights_range(1, 1280),
    drop_forgotten_tracks(true),
    max_num_objects_in_track(300),
    averaging_window_size_for_rects(1),
    averaging_window_size_for_labels(1) {}

bool IsInRange(float x, const cv::Vec2f& v) { return v[0] <= x && x <= v[1]; }
bool IsInRange(float x, float a, float b) { return a <= x && x <= b; }

void Tracker::FilterDetectionsAndStore(const TrackedObjects& detections) {
    m_detections.clear();
    for (const auto& det : detections) {
        float aspect_ratio = static_cast<float>(det.rect.height) / det.rect.width;
        if (det.confidence > m_params.min_det_conf &&
                IsInRange(aspect_ratio, m_params.bbox_aspect_ratios_range) &&
                IsInRange(static_cast<float>(det.rect.height), m_params.bbox_heights_range)) {
            m_detections.emplace_back(det);
        }
    }
}

void Tracker::SolveAssignmentProblem(
    const std::set<size_t>& track_ids, const TrackedObjects& detections,
    std::set<size_t>* unmatched_tracks, std::set<size_t>* unmatched_detections,
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
    ComputeDissimilarityMatrix(track_ids, detections, &dissimilarity);

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

bool Tracker::EraseTrackIfBBoxIsOutOfFrame(size_t track_id) {
    if (m_tracks.find(track_id) == m_tracks.end())
        return true;
    auto c = Center(m_tracks.at(track_id).back().rect);
    if (m_frame_size != cv::Size() &&
            (c.x < 0 || c.y < 0 || c.x > m_frame_size.width ||
             c.y > m_frame_size.height)) {
        m_tracks.at(track_id).lost = m_params.forget_delay + 1;
        m_active_track_ids.erase(track_id);
        return true;
    }
    return false;
}

bool Tracker::EraseTrackIfItWasLostTooManyFramesAgo(size_t track_id) {
    if (m_tracks.find(track_id) == m_tracks.end())
        return true;
    if (m_tracks.at(track_id).lost > m_params.forget_delay) {
        m_active_track_ids.erase(track_id);
        return true;
    }
    return false;
}

bool Tracker::UptateLostTrackAndEraseIfItsNeeded(size_t track_id) {
    m_tracks.at(track_id).lost++;
    bool erased = EraseTrackIfBBoxIsOutOfFrame(track_id);
    if (!erased) erased = EraseTrackIfItWasLostTooManyFramesAgo(track_id);
    return erased;
}

void Tracker::UpdateLostTracks(const std::set<size_t>& track_ids) {
    for (auto track_id : track_ids) {
        UptateLostTrackAndEraseIfItsNeeded(track_id);
    }
}

void Tracker::Process(const cv::Mat& frame, const TrackedObjects& detections, int frame_idx) {
    if (m_frame_size == cv::Size()) {
        m_frame_size = frame.size();
    } else {
        CV_Assert(m_frame_size == frame.size());
    }

    FilterDetectionsAndStore(detections);
    for (auto& obj : m_detections) {
        obj.frame_idx = frame_idx;
    }

    auto active_tracks = m_active_track_ids;

    if (!active_tracks.empty() && !m_detections.empty()) {
        std::set<size_t> unmatched_tracks, unmatched_detections;
        std::set<std::tuple<size_t, size_t, float>> matches;

        SolveAssignmentProblem(active_tracks, m_detections, &unmatched_tracks,
                               &unmatched_detections, &matches);

        for (const auto& match : matches) {
            size_t track_id = std::get<0>(match);
            size_t det_id = std::get<1>(match);
            float conf = std::get<2>(match);
            if (conf > m_params.affinity_thr) {
                AppendToTrack(track_id, m_detections[det_id]);
                unmatched_detections.erase(det_id);
            } else {
                unmatched_tracks.insert(track_id);
            }
        }

        AddNewTracks(m_detections, unmatched_detections);
        UpdateLostTracks(unmatched_tracks);

        for (size_t id : active_tracks) {
            EraseTrackIfBBoxIsOutOfFrame(id);
        }
    } else {
        AddNewTracks(m_detections);
        UpdateLostTracks(active_tracks);
    }

    if (m_params.drop_forgotten_tracks)
        DropForgottenTracks();
}

void Tracker::DropForgottenTracks() {
    std::unordered_map<size_t, Track> new_tracks;
    std::set<size_t> new_active_tracks;

    size_t max_id = 0;
    if (!m_active_track_ids.empty())
        max_id = *std::max_element(m_active_track_ids.begin(), m_active_track_ids.end());

    const size_t kMaxTrackID = 10000;
    bool reassign_id = max_id > kMaxTrackID;

    size_t counter = 0;
    for (const auto& pair : m_tracks) {
        if (!IsTrackForgotten(pair.first)) {
            new_tracks.emplace(reassign_id ? counter : pair.first, pair.second);
            new_active_tracks.emplace(reassign_id ? counter : pair.first);
            counter++;
        }
    }
    m_tracks.swap(new_tracks);
    m_active_track_ids.swap(new_active_tracks);

    m_tracks_counter = reassign_id ? counter : m_tracks_counter;
}

float Tracker::ShapeAffinity(const cv::Rect& trk, const cv::Rect& det) {
    float w_dist = static_cast<float>(std::fabs(trk.width - det.width)) / static_cast<float>(trk.width + det.width);
    float h_dist = static_cast<float>(std::fabs(trk.height - det.height)) / static_cast<float>(trk.height + det.height);
    return exp(-m_params.shape_affinity_w * (w_dist + h_dist));
}

float Tracker::MotionAffinity(const cv::Rect& trk, const cv::Rect& det) {
    float x_dist = static_cast<float>(trk.x - det.x) * (trk.x - det.x) / (det.width * det.width);
    float y_dist = static_cast<float>(trk.y - det.y) * (trk.y - det.y) / (det.height * det.height);
    return exp(-m_params.motion_affinity_w * (x_dist + y_dist));
}

void Tracker::ComputeDissimilarityMatrix(
    const std::set<size_t>& active_tracks, const TrackedObjects& detections, cv::Mat* dissimilarity_matrix) {
    dissimilarity_matrix->create(active_tracks.size(), detections.size(), CV_32F);
    size_t i = 0;
    for (auto id : active_tracks) {
        auto ptr = dissimilarity_matrix->ptr<float>(i);
        for (size_t j = 0; j < detections.size(); j++) {
            auto last_det = m_tracks.at(id).objects.back();
            ptr[j] = Distance(last_det, detections[j]);
        }
        i++;
    }
}

void Tracker::AddNewTracks(const TrackedObjects& detections) {
    for (size_t i = 0; i < detections.size(); i++) {
        AddNewTrack(detections[i]);
    }
}

void Tracker::AddNewTracks(const TrackedObjects& detections, const std::set<size_t>& ids) {
    for (size_t i : ids) {
        CV_Assert(i < detections.size());
        AddNewTrack(detections[i]);
    }
}

void Tracker::AddNewTrack(const TrackedObject& detection) {
    auto detection_with_id = detection;
    detection_with_id.object_id = m_tracks_counter;
    m_tracks.emplace(std::pair<size_t, Track>(m_tracks_counter, Track({detection_with_id})));

    m_active_track_ids.insert(m_tracks_counter);
    m_tracks_counter++;
}

void Tracker::AppendToTrack(size_t track_id, const TrackedObject& detection) {
    CV_Assert(!IsTrackForgotten(track_id));

    auto detection_with_id = detection;
    detection_with_id.object_id = track_id;

    auto& track = m_tracks.at(track_id);

    track.objects.emplace_back(detection_with_id);
    track.lost = 0;
    track.length++;

    if (m_params.max_num_objects_in_track > 0) {
        while (track.size() > static_cast<size_t>(m_params.max_num_objects_in_track)) {
            track.objects.erase(track.objects.begin());
        }
    }
}

float Tracker::Distance(const TrackedObject& obj1, const TrackedObject& obj2) {
    const float eps = 1e-6f;
    float shp_aff = ShapeAffinity(obj1.rect, obj2.rect);
    if (shp_aff < eps) return 1.0;

    float mot_aff = MotionAffinity(obj1.rect, obj2.rect);
    if (mot_aff < eps) return 1.0;

    return 1.0f - shp_aff * mot_aff;
}

bool Tracker::IsTrackValid(size_t id) const {
    const auto& track = m_tracks.at(id);
    const auto& objects = track.objects;
    if (objects.empty()) {
        return false;
    }
    size_t duration_frames = objects.back().frame_idx - track.first_object.frame_idx;
    if (duration_frames < m_params.min_track_duration)
        return false;
    return true;
}

bool Tracker::IsTrackForgotten(size_t id) const {
    return m_tracks.at(id).lost > m_params.forget_delay;
}

void Tracker::Reset() {
    m_active_track_ids.clear();
    m_tracks.clear();

    m_detections.clear();

    m_tracks_counter = 0;

    m_frame_size = cv::Size();
}

TrackedObjects Tracker::TrackedDetections() const {
    TrackedObjects detections;
    for (size_t idx : active_track_ids()) {
        auto track = tracks().at(idx);
        if (IsTrackValid(idx) && !track.lost) {
            detections.emplace_back(track.objects.back());
        }
    }
    return detections;
}

TrackedObjects Tracker::TrackedDetectionsWithLabels() const {
    TrackedObjects detections;
    for (size_t idx : active_track_ids()) {
        const auto& track = tracks().at(idx);
        if (IsTrackValid(idx) && !track.lost) {
            TrackedObject object = track.objects.back();
            int counter = 1;
            size_t start = static_cast<int>(track.objects.size()) >= m_params.averaging_window_size_for_rects ?
                        track.objects.size() - m_params.averaging_window_size_for_rects : 0;

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

            object.label = LabelWithMaxFrequencyInTrack(track, m_params.averaging_window_size_for_labels);
            object.object_id = idx;

            detections.push_back(object);
        }
    }
    return detections;
}

int LabelWithMaxFrequencyInTrack(const Track& track, int window_size) {
    std::unordered_map<int, int> frequencies;
    int max_frequent_count = 0;
    int max_frequent_id = TrackedObject::UNKNOWN_LABEL_IDX;

    int start = static_cast<int>(track.objects.size()) >= window_size ?
        static_cast<int>(track.objects.size()) - window_size : 0;

    for (size_t i = start; i < track.objects.size(); i++) {
        const auto& detection = track.objects[i];
        if (detection.label == TrackedObject::UNKNOWN_LABEL_IDX)
            continue;
        int count = ++frequencies[detection.label];
        if (count > max_frequent_count) {
            max_frequent_count = count;
            max_frequent_id = detection.label;
        }
    }
    return max_frequent_id;
}

std::vector<Track> UpdateTrackLabelsToBestAndFilterOutUnknowns(const std::vector<Track>& tracks) {
    std::vector<Track> new_tracks;
    for (auto& track : tracks) {
        int best_label = LabelWithMaxFrequencyInTrack(track, std::numeric_limits<int>::max());
        if (best_label == TrackedObject::UNKNOWN_LABEL_IDX)
            continue;

        Track new_track = track;

        for (auto& obj : new_track.objects) {
            obj.label = best_label;
        }
        new_track.first_object.label = best_label;

        new_tracks.emplace_back(std::move(new_track));
    }
    return new_tracks;
}

const std::unordered_map<size_t, Track>& Tracker::tracks() const {
    return m_tracks;
}

std::vector<Track> Tracker::vector_tracks() const {
    std::set<size_t> keys;
    for (auto& cur_pair : tracks()) {
        keys.insert(cur_pair.first);
    }
    std::vector<Track> vec_tracks;
    for (size_t k : keys) {
        vec_tracks.push_back(tracks().at(k));
    }
    return vec_tracks;
}
