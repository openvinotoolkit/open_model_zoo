// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <map>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <utility>
#include <limits>
#include <algorithm>

#include "core.hpp"
#include "tracker.hpp"
#include "utils.hpp"
#include "kuhn_munkres.hpp"

namespace {
cv::Point Center(const cv::Rect& rect) {
    return cv::Point(rect.x + rect.width * .5, rect.y + rect.height * .5);
}

std::vector<cv::Point> Centers(const TrackedObjects &detections) {
    std::vector<cv::Point> centers(detections.size());
    for (size_t i = 0; i < detections.size(); i++) {
        centers[i] = Center(detections[i].rect);
    }
    return centers;
}

DetectionLog ConvertTracksToDetectionLog(const ObjectTracks& tracks) {
    DetectionLog log;

    // Combine detected objects by respective frame indices.
    std::map<int, TrackedObjects> objects;
    for (const auto& track : tracks)
        for (const auto& object : track.second) {
            auto itr = objects.find(object.frame_idx);
            if (itr != objects.end())
                itr->second.emplace_back(object);
            else
                objects.emplace(object.frame_idx, TrackedObjects{object});
        }

    for (const auto& frame_res : objects) {
        DetectionLogEntry entry;
        entry.frame_idx = frame_res.first;
        entry.objects = std::move(frame_res.second);
        log.push_back(std::move(entry));
    }

    return log;
}

inline bool IsInRange(float val, float min, float max) {
    return min <= val && val <= max;
}

inline bool IsInRange(float val, cv::Vec2f range) {
    return IsInRange(val, range[0], range[1]);
}

std::vector<cv::Scalar> GenRandomColors(int colors_num) {
    std::vector<cv::Scalar> colors(colors_num);
    for (int i = 0; i < colors_num; i++) {
        colors[i] = cv::Scalar(static_cast<uchar>(255. * rand() / RAND_MAX),  // NOLINT
                               static_cast<uchar>(255. * rand() / RAND_MAX),  // NOLINT
                               static_cast<uchar>(255. * rand() / RAND_MAX));  // NOLINT
    }
    return colors;
}

}  // anonymous namespace

TrackerParams::TrackerParams()
    : min_track_duration(1000),
    forget_delay(150),
    aff_thr_fast(0.8),
    aff_thr_strong(0.75),
    shape_affinity_w(0.5),
    motion_affinity_w(0.2),
    time_affinity_w(0.0),
    min_det_conf(0.65),
    bbox_aspect_ratios_range(0.666, 5.0),
    bbox_heights_range(40, 1000),
    predict(25),
    strong_affinity_thr(0.2805),
    reid_thr(0.61),
    drop_forgotten_tracks(true),
    max_num_objects_in_track(300) {}

void ValidateParams(const TrackerParams &p) {
    PT_CHECK_GE(p.min_track_duration, static_cast<size_t>(500));
    PT_CHECK_LE(p.min_track_duration, static_cast<size_t>(10000));

    PT_CHECK_LE(p.forget_delay, static_cast<size_t>(10000));

    PT_CHECK_GE(p.aff_thr_fast, 0.0f);
    PT_CHECK_LE(p.aff_thr_fast, 1.0f);

    PT_CHECK_GE(p.aff_thr_strong, 0.0f);
    PT_CHECK_LE(p.aff_thr_strong, 1.0f);

    PT_CHECK_GE(p.shape_affinity_w, 0.0f);
    PT_CHECK_LE(p.shape_affinity_w, 100.0f);

    PT_CHECK_GE(p.motion_affinity_w, 0.0f);
    PT_CHECK_LE(p.motion_affinity_w, 100.0f);

    PT_CHECK_GE(p.time_affinity_w, 0.0f);
    PT_CHECK_LE(p.time_affinity_w, 100.0f);

    PT_CHECK_GE(p.min_det_conf, 0.0f);
    PT_CHECK_LE(p.min_det_conf, 1.0f);

    PT_CHECK_GE(p.bbox_aspect_ratios_range[0], 0.0f);
    PT_CHECK_LE(p.bbox_aspect_ratios_range[1], 10.0f);
    PT_CHECK_LT(p.bbox_aspect_ratios_range[0], p.bbox_aspect_ratios_range[1]);

    PT_CHECK_GE(p.bbox_heights_range[0], 10.0f);
    PT_CHECK_LE(p.bbox_heights_range[1], 1080.0f);
    PT_CHECK_LT(p.bbox_heights_range[0], p.bbox_heights_range[1]);

    PT_CHECK_GE(p.predict, 0);
    PT_CHECK_LE(p.predict, 10000);

    PT_CHECK_GE(p.strong_affinity_thr, 0.0f);
    PT_CHECK_LE(p.strong_affinity_thr, 1.0f);

    PT_CHECK_GE(p.reid_thr, 0.0f);
    PT_CHECK_LE(p.reid_thr, 1.0f);


    if (p.max_num_objects_in_track > 0) {
        int min_required_track_length = static_cast<int>(p.forget_delay);
        PT_CHECK_GE(p.max_num_objects_in_track, min_required_track_length);
        PT_CHECK_LE(p.max_num_objects_in_track, 10000);
    }
}

// Returns confusion matrix as:
//   |tp fn|
//   |fp tn|
cv::Mat PedestrianTracker::ConfusionMatrix(const std::vector<Match> &matches) {
    const bool kNegative = false;
    cv::Mat conf_mat(2, 2, CV_32F, cv::Scalar(0));
    for (const auto &m : matches) {
        conf_mat.at<float>(m.gt_label == kNegative, m.pr_label == kNegative)++;
    }

    return conf_mat;
}

PedestrianTracker::PedestrianTracker(const TrackerParams &params)
    : params_(params),
    descriptor_strong_(nullptr),
    distance_strong_(nullptr),
    collect_matches_(true),
    tracks_counter_(0),
    valid_tracks_counter_(0),
    frame_size_(0, 0),
    prev_timestamp_(std::numeric_limits<uint64_t>::max()) {
        ValidateParams(params);
    }

// Pipeline parameters getter.
const TrackerParams &PedestrianTracker::params() const { return params_; }

// Pipeline parameters setter.
void PedestrianTracker::set_params(const TrackerParams &params) {
    ValidateParams(params);
    params_ = params;
}

// Descriptor fast getter.
const PedestrianTracker::Descriptor &PedestrianTracker::descriptor_fast() const {
    return descriptor_fast_;
}

// Descriptor fast setter.
void PedestrianTracker::set_descriptor_fast(const Descriptor &val) {
    descriptor_fast_ = val;
}

// Descriptor strong getter.
const PedestrianTracker::Descriptor &PedestrianTracker::descriptor_strong() const {
    return descriptor_strong_;
}

// Descriptor strong setter.
void PedestrianTracker::set_descriptor_strong(const Descriptor &val) {
    descriptor_strong_ = val;
}

// Distance fast getter.
const PedestrianTracker::Distance &PedestrianTracker::distance_fast() const { return distance_fast_; }

// Distance fast setter.
void PedestrianTracker::set_distance_fast(const Distance &val) { distance_fast_ = val; }

// Distance strong getter.
const PedestrianTracker::Distance &PedestrianTracker::distance_strong() const { return distance_strong_; }

// Distance strong setter.
void PedestrianTracker::set_distance_strong(const Distance &val) { distance_strong_ = val; }

// Returns all tracks including forgotten (lost too many frames ago).
const std::unordered_map<size_t, Track> &
PedestrianTracker::tracks() const {
    return tracks_;
}

// Returns indexes of active tracks only.
const std::set<size_t> &PedestrianTracker::active_track_ids() const {
    return active_track_ids_;
}


// Returns detection log which is used for tracks saving.
DetectionLog PedestrianTracker::GetDetectionLog(const bool valid_only) const {
    return ConvertTracksToDetectionLog(all_tracks(valid_only));
}

// Returns decisions made by heuristic based on fast distance/descriptor and
// shape, motion and time affinity.
const std::vector<PedestrianTracker::Match> &
PedestrianTracker::base_classifier_matches() const {
    return base_classifier_matches_;
}

// Returns decisions made by heuristic based on strong distance/descriptor
// and
// shape, motion and time affinity.
const std::vector<PedestrianTracker::Match> &PedestrianTracker::reid_based_classifier_matches() const {
    return reid_based_classifier_matches_;
}

// Returns decisions made by strong distance/descriptor affinity.
const std::vector<PedestrianTracker::Match> &PedestrianTracker::reid_classifier_matches() const {
    return reid_classifier_matches_;
}

TrackedObjects PedestrianTracker::FilterDetections(
    const TrackedObjects &detections) const {
    TrackedObjects filtered_detections;
    for (const auto &det : detections) {
        float aspect_ratio = static_cast<float>(det.rect.height) / det.rect.width;
        if (det.confidence > params_.min_det_conf &&
            IsInRange(aspect_ratio, params_.bbox_aspect_ratios_range) &&
            IsInRange(det.rect.height, params_.bbox_heights_range)) {
            filtered_detections.emplace_back(det);
        }
    }
    return filtered_detections;
}

void PedestrianTracker::SolveAssignmentProblem(
    const std::set<size_t> &track_ids, const TrackedObjects &detections,
    const std::vector<cv::Mat> &descriptors, float thr,
    std::set<size_t> *unmatched_tracks, std::set<size_t> *unmatched_detections,
    std::set<std::tuple<size_t, size_t, float>> *matches) {
    PT_CHECK(unmatched_tracks);
    PT_CHECK(unmatched_detections);
    unmatched_tracks->clear();
    unmatched_detections->clear();

    PT_CHECK(!track_ids.empty());
    PT_CHECK(!detections.empty());
    PT_CHECK(descriptors.size() == detections.size());
    PT_CHECK(matches);
    matches->clear();

    cv::Mat dissimilarity;
    ComputeDissimilarityMatrix(track_ids, detections, descriptors,
                               &dissimilarity);

    auto res = KuhnMunkres().Solve(dissimilarity);

    for (size_t i = 0; i < detections.size(); i++) {
        unmatched_detections->insert(i);
    }

    constexpr float kMinIOUForPossibleDuplicates = 0.3;

    size_t i = 0;
    for (auto id : track_ids) {
        if (res[i] < detections.size()) {
            matches->emplace(id, res[i], 1 - dissimilarity.at<float>(i, res[i]));
        } else {
            unmatched_tracks->insert(id);
        }
        i++;
    }
}

const ObjectTracks PedestrianTracker::all_tracks(bool valid_only) const {
    ObjectTracks all_objects;
    size_t counter = 0;

    std::set<size_t> sorted_ids;
    for (const auto &pair : tracks()) {
        sorted_ids.emplace(pair.first);
    }

    for (size_t id : sorted_ids) {
        if (!valid_only || IsTrackValid(id)) {
            TrackedObjects filtered_objects;
            for (const auto &object : tracks().at(id).objects) {
                filtered_objects.emplace_back(object);
                filtered_objects.back().object_id = counter;
            }
            all_objects.emplace(counter++, filtered_objects);
        }
    }
    return all_objects;
}

cv::Rect PedestrianTracker::PredictRect(size_t id, size_t k,
                                        size_t s) const {
    const auto &track = tracks_.at(id);
    PT_CHECK(!track.empty());

    if (track.size() == 1) {
        return track[0].rect;
    }

    size_t start_i = track.size() > k ? track.size() - k : 0;
    float width = 0, height = 0;

    for (size_t i = start_i; i < track.size(); i++) {
        width += track[i].rect.width;
        height += track[i].rect.height;
    }

    PT_CHECK(track.size() - start_i > 0);
    width /= (track.size() - start_i);
    height /= (track.size() - start_i);

    float delim = 0;
    cv::Point2f d(0, 0);

    for (size_t i = start_i + 1; i < track.size(); i++) {
        d += cv::Point2f(Center(track[i].rect) - Center(track[i - 1].rect));
        delim += (track[i].frame_idx - track[i - 1].frame_idx);
    }

    if (delim) {
        d /= delim;
    }

    s += 1;

    cv::Point c = Center(track.back().rect);
    return cv::Rect(c.x - width / 2 + d.x * s, c.y - height / 2 + d.y * s, width,
                    height);
}


bool PedestrianTracker::EraseTrackIfBBoxIsOutOfFrame(size_t track_id) {
    if (tracks_.find(track_id) == tracks_.end()) return true;
    auto c = Center(tracks_.at(track_id).predicted_rect);
    if (!prev_frame_size_.empty() &&
        (c.x < 0 || c.y < 0 || c.x > prev_frame_size_.width ||
         c.y > prev_frame_size_.height)) {
        tracks_.at(track_id).lost = params_.forget_delay + 1;
        for (auto id : active_track_ids()) {
            size_t min_id = std::min(id, track_id);
            size_t max_id = std::max(id, track_id);
            tracks_dists_.erase(std::pair<size_t, size_t>(min_id, max_id));
        }
        active_track_ids_.erase(track_id);
        return true;
    }
    return false;
}

bool PedestrianTracker::EraseTrackIfItWasLostTooManyFramesAgo(
    size_t track_id) {
    if (tracks_.find(track_id) == tracks_.end()) return true;
    if (tracks_.at(track_id).lost > params_.forget_delay) {
        for (auto id : active_track_ids()) {
            size_t min_id = std::min(id, track_id);
            size_t max_id = std::max(id, track_id);
            tracks_dists_.erase(std::pair<size_t, size_t>(min_id, max_id));
        }
        active_track_ids_.erase(track_id);

        return true;
    }
    return false;
}

bool PedestrianTracker::UpdateLostTrackAndEraseIfItsNeeded(
    size_t track_id) {
    tracks_.at(track_id).lost++;
    tracks_.at(track_id).predicted_rect =
        PredictRect(track_id, params().predict, tracks_.at(track_id).lost);

    bool erased = EraseTrackIfBBoxIsOutOfFrame(track_id);
    if (!erased) erased = EraseTrackIfItWasLostTooManyFramesAgo(track_id);
    return erased;
}

void PedestrianTracker::UpdateLostTracks(
    const std::set<size_t> &track_ids) {
    for (auto track_id : track_ids) {
        UpdateLostTrackAndEraseIfItsNeeded(track_id);
    }
}

void PedestrianTracker::Process(const cv::Mat &frame,
                                const TrackedObjects &input_detections,
                                uint64_t timestamp) {
    if (prev_timestamp_ != std::numeric_limits<uint64_t>::max())
        PT_CHECK_LT(prev_timestamp_, timestamp);

    if (frame_size_ == cv::Size(0, 0)) {
        frame_size_ = frame.size();
    } else {
        PT_CHECK_EQ(frame_size_, frame.size());
    }

    TrackedObjects detections = FilterDetections(input_detections);
    for (auto &obj : detections) {
        obj.timestamp = timestamp;
    }

    std::vector<cv::Mat> descriptors_fast;
    ComputeFastDesciptors(frame, detections, &descriptors_fast);

    auto active_tracks = active_track_ids_;

    if (!active_tracks.empty() && !detections.empty()) {
        std::set<size_t> unmatched_tracks, unmatched_detections;
        std::set<std::tuple<size_t, size_t, float>> matches;

        SolveAssignmentProblem(active_tracks, detections, descriptors_fast,
                               params_.aff_thr_fast, &unmatched_tracks,
                               &unmatched_detections, &matches);

        std::map<size_t, std::pair<bool, cv::Mat>> is_matching_to_track;

        if (distance_strong_) {
            std::vector<std::pair<size_t, size_t>> reid_track_and_det_ids =
                GetTrackToDetectionIds(matches);
            is_matching_to_track = StrongMatching(
                frame, detections, reid_track_and_det_ids);
        }

        for (const auto &match : matches) {
            size_t track_id = std::get<0>(match);
            size_t det_id = std::get<1>(match);
            float conf = std::get<2>(match);

            auto last_det = tracks_.at(track_id).objects.back();
            last_det.rect = tracks_.at(track_id).predicted_rect;

            if (collect_matches_ && last_det.object_id >= 0 &&
                detections[det_id].object_id >= 0) {
                base_classifier_matches_.emplace_back(
                    tracks_.at(track_id).objects.back(), last_det.rect,
                    detections[det_id], conf > params_.aff_thr_fast);
            }

            if (conf > params_.aff_thr_fast) {
                AppendToTrack(frame, track_id, detections[det_id],
                              descriptors_fast[det_id], cv::Mat());
                unmatched_detections.erase(det_id);
            } else {
                if (conf > params_.strong_affinity_thr) {
                    if (distance_strong_ && is_matching_to_track[track_id].first) {
                        AppendToTrack(frame, track_id, detections[det_id],
                                      descriptors_fast[det_id],
                                      is_matching_to_track[track_id].second.clone());
                    } else {
                        if (UpdateLostTrackAndEraseIfItsNeeded(track_id)) {
                            AddNewTrack(frame, detections[det_id], descriptors_fast[det_id],
                                        distance_strong_
                                        ? is_matching_to_track[track_id].second.clone()
                                        : cv::Mat());
                        }
                    }

                    unmatched_detections.erase(det_id);
                } else {
                    unmatched_tracks.insert(track_id);
                }
            }
        }

        AddNewTracks(frame, detections, descriptors_fast, unmatched_detections);
        UpdateLostTracks(unmatched_tracks);

        for (size_t id : active_tracks) {
            EraseTrackIfBBoxIsOutOfFrame(id);
        }
    } else {
        AddNewTracks(frame, detections, descriptors_fast);
        UpdateLostTracks(active_tracks);
    }

    prev_frame_size_ = frame.size();
    if (params_.drop_forgotten_tracks) DropForgottenTracks();

    tracks_dists_.clear();
    prev_timestamp_ = timestamp;
}

void PedestrianTracker::DropForgottenTracks() {
    std::unordered_map<size_t, Track> new_tracks;
    std::set<size_t> new_active_tracks;

    size_t max_id = 0;
    if (!active_track_ids_.empty())
        max_id =
            *std::max_element(active_track_ids_.begin(), active_track_ids_.end());

    const size_t kMaxTrackID = 10000;
    bool reassign_id = max_id > kMaxTrackID;

    size_t counter = 0;
    for (const auto &pair : tracks_) {
        if (!IsTrackForgotten(pair.first)) {
            new_tracks.emplace(reassign_id ? counter : pair.first, pair.second);
            new_active_tracks.emplace(reassign_id ? counter : pair.first);
            counter++;

        } else {
            if (IsTrackValid(pair.first)) {
                valid_tracks_counter_++;
            }
        }
    }
    tracks_.swap(new_tracks);
    active_track_ids_.swap(new_active_tracks);

    tracks_counter_ = reassign_id ? counter : tracks_counter_;
}

void PedestrianTracker::DropForgottenTrack(size_t track_id) {
    PT_CHECK(IsTrackForgotten(track_id));
    PT_CHECK(active_track_ids_.count(track_id) == 0);
    tracks_.erase(track_id);
}

float PedestrianTracker::ShapeAffinity(float weight, const cv::Rect &trk,
                                       const cv::Rect &det) {
    float w_dist = std::fabs(trk.width - det.width) / (trk.width + det.width);
    float h_dist = std::fabs(trk.height - det.height) / (trk.height + det.height);
    return exp(-weight * (w_dist + h_dist));
}

float PedestrianTracker::MotionAffinity(float weight, const cv::Rect &trk,
                                        const cv::Rect &det) {
    float x_dist = static_cast<float>(trk.x - det.x) * (trk.x - det.x) /
        (det.width * det.width);
    float y_dist = static_cast<float>(trk.y - det.y) * (trk.y - det.y) /
        (det.height * det.height);
    return exp(-weight * (x_dist + y_dist));
}

float PedestrianTracker::TimeAffinity(float weight, const float &trk_time,
                                      const float &det_time) {
    return exp(-weight * std::fabs(trk_time - det_time));
}

void PedestrianTracker::ComputeFastDesciptors(
    const cv::Mat &frame, const TrackedObjects &detections,
    std::vector<cv::Mat> *desriptors) {
    *desriptors = std::vector<cv::Mat>(detections.size(), cv::Mat());
    for (size_t i = 0; i < detections.size(); i++) {
        descriptor_fast_->Compute(frame(detections[i].rect).clone(),
                                  &((*desriptors)[i]));
    }
}

void PedestrianTracker::ComputeDissimilarityMatrix(
    const std::set<size_t> &active_tracks, const TrackedObjects &detections,
    const std::vector<cv::Mat> &descriptors_fast,
    cv::Mat *dissimilarity_matrix) {
    cv::Mat am(active_tracks.size(), detections.size(), CV_32F, cv::Scalar(0));
    size_t i = 0;
    for (auto id : active_tracks) {
        auto ptr = am.ptr<float>(i);
        for (size_t j = 0; j < descriptors_fast.size(); j++) {
            auto last_det = tracks_.at(id).objects.back();
            last_det.rect = tracks_.at(id).predicted_rect;
            ptr[j] = AffinityFast(tracks_.at(id).descriptor_fast, last_det,
                                  descriptors_fast[j], detections[j]);
        }
        i++;
    }
    *dissimilarity_matrix = 1.0 - am;
}

std::vector<float> PedestrianTracker::ComputeDistances(
    const cv::Mat &frame,
    const TrackedObjects& detections,
    const std::vector<std::pair<size_t, size_t>> &track_and_det_ids,
    std::map<size_t, cv::Mat> *det_id_to_descriptor) {
    std::map<size_t, size_t> det_to_batch_ids;
    std::map<size_t, size_t> track_to_batch_ids;

    std::vector<cv::Mat> images;
    std::vector<cv::Mat> descriptors;
    for (size_t i = 0; i < track_and_det_ids.size(); i++) {
        size_t track_id = track_and_det_ids[i].first;
        size_t det_id = track_and_det_ids[i].second;

        if (tracks_.at(track_id).descriptor_strong.empty()) {
            images.push_back(tracks_.at(track_id).last_image);
            descriptors.push_back(cv::Mat());
            track_to_batch_ids[track_id] = descriptors.size() - 1;
        }

        images.push_back(frame(detections[det_id].rect));
        descriptors.push_back(cv::Mat());
        det_to_batch_ids[det_id] = descriptors.size() - 1;
    }

    descriptor_strong_->Compute(images, &descriptors);

    std::vector<cv::Mat> descriptors1;
    std::vector<cv::Mat> descriptors2;
    for (size_t i = 0; i < track_and_det_ids.size(); i++) {
        size_t track_id = track_and_det_ids[i].first;
        size_t det_id = track_and_det_ids[i].second;

        if (tracks_.at(track_id).descriptor_strong.empty()) {
            tracks_.at(track_id).descriptor_strong =
                descriptors[track_to_batch_ids[track_id]].clone();
        }
        (*det_id_to_descriptor)[det_id] = descriptors[det_to_batch_ids[det_id]];

        descriptors1.push_back(descriptors[det_to_batch_ids[det_id]]);
        descriptors2.push_back(tracks_.at(track_id).descriptor_strong);
    }

    std::vector<float> distances =
        distance_strong_->Compute(descriptors1, descriptors2);

    return distances;
}

std::vector<std::pair<size_t, size_t>>
PedestrianTracker::GetTrackToDetectionIds(
    const std::set<std::tuple<size_t, size_t, float>> &matches) {
    std::vector<std::pair<size_t, size_t>> track_and_det_ids;

    for (const auto &match : matches) {
        size_t track_id = std::get<0>(match);
        size_t det_id = std::get<1>(match);
        float conf = std::get<2>(match);
        if (conf < params_.aff_thr_fast && conf > params_.strong_affinity_thr) {
            track_and_det_ids.emplace_back(track_id, det_id);
        }
    }
    return track_and_det_ids;
}

std::map<size_t, std::pair<bool, cv::Mat>>
PedestrianTracker::StrongMatching(
    const cv::Mat &frame,
    const TrackedObjects& detections,
    const std::vector<std::pair<size_t, size_t>> &track_and_det_ids) {
    std::map<size_t, std::pair<bool, cv::Mat>> is_matching;

    if (track_and_det_ids.size() == 0) {
        return is_matching;
    }

    std::map<size_t, cv::Mat> det_ids_to_descriptors;
    std::vector<float> distances =
        ComputeDistances(frame, detections,
                         track_and_det_ids, &det_ids_to_descriptors);

    for (size_t i = 0; i < track_and_det_ids.size(); i++) {
        auto reid_affinity = 1.0 - distances[i];

        size_t track_id = track_and_det_ids[i].first;
        size_t det_id = track_and_det_ids[i].second;

        const auto& track = tracks_.at(track_id);
        const auto& detection = detections[det_id];

        auto last_det = track.objects.back();
        last_det.rect = track.predicted_rect;

        float affinity = reid_affinity * Affinity(last_det, detection);

        if (collect_matches_ && last_det.object_id >= 0 &&
            detection.object_id >= 0) {
            reid_classifier_matches_.emplace_back(track.objects.back(), last_det.rect,
                                                  detection,
                                                  reid_affinity > params_.reid_thr);

            reid_based_classifier_matches_.emplace_back(
                track.objects.back(), last_det.rect, detection,
                affinity > params_.aff_thr_strong);
        }

        bool is_detection_matching =
            reid_affinity > params_.reid_thr && affinity > params_.aff_thr_strong;

        is_matching[track_id] = std::pair<bool, cv::Mat>(
            is_detection_matching, det_ids_to_descriptors[det_id]);
    }
    return is_matching;
}

void PedestrianTracker::AddNewTracks(
    const cv::Mat &frame, const TrackedObjects &detections,
    const std::vector<cv::Mat> &descriptors_fast) {
    PT_CHECK(detections.size() == descriptors_fast.size());
    for (size_t i = 0; i < detections.size(); i++) {
        AddNewTrack(frame, detections[i], descriptors_fast[i]);
    }
}

void PedestrianTracker::AddNewTracks(
    const cv::Mat &frame, const TrackedObjects &detections,
    const std::vector<cv::Mat> &descriptors_fast, const std::set<size_t> &ids) {
    PT_CHECK(detections.size() == descriptors_fast.size());
    for (size_t i : ids) {
        PT_CHECK(i < detections.size());
        AddNewTrack(frame, detections[i], descriptors_fast[i]);
    }
}

void PedestrianTracker::AddNewTrack(const cv::Mat &frame,
                                    const TrackedObject &detection,
                                    const cv::Mat &descriptor_fast,
                                    const cv::Mat &descriptor_strong) {
    auto detection_with_id = detection;
    detection_with_id.object_id = tracks_counter_;
    tracks_.emplace(std::pair<size_t, Track>(
            tracks_counter_,
            Track({detection_with_id}, frame(detection.rect).clone(),
                  descriptor_fast.clone(), descriptor_strong.clone())));

    for (size_t id : active_track_ids_) {
        tracks_dists_.emplace(std::pair<size_t, size_t>(id, tracks_counter_),
                              std::numeric_limits<float>::max());
    }

    active_track_ids_.insert(tracks_counter_);
    tracks_counter_++;
}

void PedestrianTracker::AppendToTrack(const cv::Mat &frame,
                                      size_t track_id,
                                      const TrackedObject &detection,
                                      const cv::Mat &descriptor_fast,
                                      const cv::Mat &descriptor_strong) {
    PT_CHECK(!IsTrackForgotten(track_id));

    auto detection_with_id = detection;
    detection_with_id.object_id = track_id;

    auto &cur_track = tracks_.at(track_id);
    cur_track.objects.emplace_back(detection_with_id);
    cur_track.predicted_rect = detection.rect;
    cur_track.lost = 0;
    cur_track.last_image = frame(detection.rect).clone();
    cur_track.descriptor_fast = descriptor_fast.clone();
    cur_track.length++;

    if (cur_track.descriptor_strong.empty()) {
        cur_track.descriptor_strong = descriptor_strong.clone();
    } else if (!descriptor_strong.empty()) {
        cur_track.descriptor_strong =
            0.5 * (descriptor_strong + cur_track.descriptor_strong);
    }


    if (params_.max_num_objects_in_track > 0) {
        while (cur_track.size() >
               static_cast<size_t>(params_.max_num_objects_in_track)) {
            cur_track.objects.erase(cur_track.objects.begin());
        }
    }
}

float PedestrianTracker::AffinityFast(const cv::Mat &descriptor1,
                                      const TrackedObject &obj1,
                                      const cv::Mat &descriptor2,
                                      const TrackedObject &obj2) {
    const float eps = 1e-6;
    float shp_aff = ShapeAffinity(params_.shape_affinity_w, obj1.rect, obj2.rect);
    if (shp_aff < eps) return 0.0;

    float mot_aff =
        MotionAffinity(params_.motion_affinity_w, obj1.rect, obj2.rect);
    if (mot_aff < eps) return 0.0;
    float time_aff =
        TimeAffinity(params_.time_affinity_w, obj1.frame_idx, obj2.frame_idx);

    if (time_aff < eps) return 0.0;

    float app_aff = 1.0 - distance_fast_->Compute(descriptor1, descriptor2);

    return shp_aff * mot_aff * app_aff * time_aff;
}

float PedestrianTracker::Affinity(const TrackedObject &obj1,
                                  const TrackedObject &obj2) {
    float shp_aff = ShapeAffinity(params_.shape_affinity_w, obj1.rect, obj2.rect);
    float mot_aff =
        MotionAffinity(params_.motion_affinity_w, obj1.rect, obj2.rect);
    float time_aff =
        TimeAffinity(params_.time_affinity_w, obj1.frame_idx, obj2.frame_idx);
    return shp_aff * mot_aff * time_aff;
}

bool PedestrianTracker::IsTrackValid(size_t id) const {
    const auto& track = tracks_.at(id);
    const auto &objects = track.objects;
    if (objects.empty()) {
        return false;
    }
    int64_t duration_ms = objects.back().timestamp - track.first_object.timestamp;
    if (duration_ms < static_cast<int64_t>(params_.min_track_duration))
        return false;
    return true;
}

bool PedestrianTracker::IsTrackForgotten(size_t id) const {
    return IsTrackForgotten(tracks_.at(id));
}

bool PedestrianTracker::IsTrackForgotten(const Track &track) const {
    return (track.lost > params_.forget_delay);
}

size_t PedestrianTracker::Count() const {
    size_t count = valid_tracks_counter_;
    for (const auto &pair : tracks_) {
        count += (IsTrackValid(pair.first) ? 1 : 0);
    }
    return count;
}

std::unordered_map<size_t, std::vector<cv::Point>>
PedestrianTracker::GetActiveTracks() const {
    std::unordered_map<size_t, std::vector<cv::Point>> active_tracks;
    for (size_t idx : active_track_ids()) {
        auto track = tracks().at(idx);
        if (IsTrackValid(idx) && !IsTrackForgotten(idx)) {
            active_tracks.emplace(idx, Centers(track.objects));
        }
    }
    return active_tracks;
}

TrackedObjects PedestrianTracker::TrackedDetections() const {
    TrackedObjects detections;
    for (size_t idx : active_track_ids()) {
        auto track = tracks().at(idx);
        if (IsTrackValid(idx) && !track.lost) {
            detections.emplace_back(track.objects.back());
        }
    }
    return detections;
}

cv::Mat PedestrianTracker::DrawActiveTracks(const cv::Mat &frame) {
    cv::Mat out_frame = frame.clone();

    if (colors_.empty()) {
        int num_colors = 100;
        colors_ = GenRandomColors(num_colors);
    }

    auto active_tracks = GetActiveTracks();
    for (auto active_track : active_tracks) {
        size_t idx = active_track.first;
        auto centers = active_track.second;
        DrawPolyline(centers, colors_[idx % colors_.size()], &out_frame);
        std::stringstream ss;
        ss << idx;
        cv::putText(out_frame, ss.str(), centers.back(), cv::FONT_HERSHEY_SCRIPT_COMPLEX, 2.0,
                    colors_[idx % colors_.size()], 3);
        auto track = tracks().at(idx);
        if (track.lost) {
            cv::line(out_frame, active_track.second.back(),
                     Center(track.predicted_rect), cv::Scalar(0, 0, 0), 4);
        }
    }

    return out_frame;
}

const cv::Size kMinFrameSize = cv::Size(320, 240);
const cv::Size kMaxFrameSize = cv::Size(1920, 1080);

void PedestrianTracker::PrintConfusionMatrices() const {
    std::cout << "Base classifier quality: " << std::endl;
    {
        auto cm = ConfusionMatrix(base_classifier_matches());
        std::cout << cm << std::endl;
        std::cout << "or" << std::endl;
        cm.row(0) = cm.row(0) / std::max(1.0, cv::sum(cm.row(0))[0]);
        cm.row(1) = cm.row(1) / std::max(1.0, cv::sum(cm.row(1))[0]);
        std::cout << cm << std::endl << std::endl;
    }

    std::cout << "Reid-based classifier quality: " << std::endl;
    {
        auto cm = ConfusionMatrix(reid_based_classifier_matches());
        std::cout << cm << std::endl;
        std::cout << "or" << std::endl;
        cm.row(0) = cm.row(0) / std::max(1.0, cv::sum(cm.row(0))[0]);
        cm.row(1) = cm.row(1) / std::max(1.0, cv::sum(cm.row(1))[0]);
        std::cout << cm << std::endl << std::endl;
    }

    std::cout << "Reid only classifier quality: " << std::endl;
    {
        auto cm = ConfusionMatrix(reid_classifier_matches());
        std::cout << cm << std::endl;
        std::cout << "or" << std::endl;
        cm.row(0) = cm.row(0) / std::max(1.0, cv::sum(cm.row(0))[0]);
        cm.row(1) = cm.row(1) / std::max(1.0, cv::sum(cm.row(1))[0]);
        std::cout << cm << std::endl << std::endl;
    }
}

void PedestrianTracker::PrintReidPerformanceCounts() const {
    if (descriptor_strong_) {
        descriptor_strong_->PrintPerformanceCounts();
    }
}
