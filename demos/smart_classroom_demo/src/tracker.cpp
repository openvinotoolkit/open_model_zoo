// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tracker.hpp"
#include <unordered_map>
#include <algorithm>
#include <utility>
#include <limits>
#include <memory>
#include <vector>
#include <tuple>
#include <set>
#include "logger.hpp"

const int TrackedObject::UNKNOWN_LABEL_IDX = -1;

class KuhnMunkres::Impl {
public:
    Impl() : n_() {}

    std::vector<size_t> Solve(const cv::Mat &dissimilarity_matrix) {
        CV_Assert(dissimilarity_matrix.type() == CV_32F);
        double min_val;
        cv::minMaxLoc(dissimilarity_matrix, &min_val);
        CV_Assert(min_val >= 0);

        n_ = std::max(dissimilarity_matrix.rows, dissimilarity_matrix.cols);
        dm_ = cv::Mat(n_, n_, CV_32F, cv::Scalar(0));
        marked_ = cv::Mat(n_, n_, CV_8S, cv::Scalar(0));
        points_ = std::vector<cv::Point>(n_ * 2);

        dissimilarity_matrix.copyTo(dm_(
                                        cv::Rect(0, 0, dissimilarity_matrix.cols, dissimilarity_matrix.rows)));

        is_row_visited_ = std::vector<int>(n_, 0);
        is_col_visited_ = std::vector<int>(n_, 0);

        Run();

        std::vector<size_t> results(dissimilarity_matrix.rows, -1);
        for (int i = 0; i < dissimilarity_matrix.rows; i++) {
            const auto ptr = marked_.ptr<char>(i);
            for (int j = 0; j < dissimilarity_matrix.cols; j++) {
                if (ptr[j] == kStar) {
                    results[i] = j;
                }
            }
        }
        return results;
    }

    void TrySimpleCase() {
        auto is_row_visited = std::vector<int>(n_, 0);
        auto is_col_visited = std::vector<int>(n_, 0);

        for (int row = 0; row < n_; row++) {
            auto ptr = dm_.ptr<float>(row);
            auto marked_ptr = marked_.ptr<char>(row);
            auto min_val = *std::min_element(ptr, ptr + n_);
            for (int col = 0; col < n_; col++) {
                ptr[col] -= min_val;
                if (ptr[col] == 0 && !is_col_visited[col] && !is_row_visited[row]) {
                    marked_ptr[col] = kStar;
                    is_col_visited[col] = 1;
                    is_row_visited[row] = 1;
                }
            }
        }
    }

    bool CheckIfOptimumIsFound() {
        int count = 0;
        for (int i = 0; i < n_; i++) {
            const auto marked_ptr = marked_.ptr<char>(i);
            for (int j = 0; j < n_; j++) {
                if (marked_ptr[j] == kStar) {
                    is_col_visited_[j] = 1;
                    count++;
                }
            }
        }

        return count >= n_;
    }

    cv::Point FindUncoveredMinValPos() {
        auto min_val = std::numeric_limits<float>::max();
        cv::Point min_val_pos(-1, -1);
        for (int i = 0; i < n_; i++) {
            if (!is_row_visited_[i]) {
                auto dm_ptr = dm_.ptr<float>(i);
                for (int j = 0; j < n_; j++) {
                    if (!is_col_visited_[j] && dm_ptr[j] < min_val) {
                        min_val = dm_ptr[j];
                        min_val_pos = cv::Point(j, i);
                    }
                }
            }
        }
        return min_val_pos;
    }

    void UpdateDissimilarityMatrix(float val) {
        for (int i = 0; i < n_; i++) {
            auto dm_ptr = dm_.ptr<float>(i);
            for (int j = 0; j < n_; j++) {
                if (is_row_visited_[i]) dm_ptr[j] += val;
                if (!is_col_visited_[j]) dm_ptr[j] -= val;
            }
        }
    }

    int FindInRow(int row, int what) {
        for (int j = 0; j < n_; j++) {
            if (marked_.at<char>(row, j) == what) {
                return j;
            }
        }
        return -1;
    }

    int FindInCol(int col, int what) {
        for (int i = 0; i < n_; i++) {
            if (marked_.at<char>(i, col) == what) {
                return i;
            }
        }
        return -1;
    }

    void Run() {
        TrySimpleCase();
        while (!CheckIfOptimumIsFound()) {
            while (true) {
                auto point = FindUncoveredMinValPos();
                auto min_val = dm_.at<float>(point.y, point.x);
                if (min_val > 0) {
                    UpdateDissimilarityMatrix(min_val);
                } else {
                    marked_.at<char>(point.y, point.x) = kPrime;
                    int col = FindInRow(point.y, kStar);
                    if (col >= 0) {
                        is_row_visited_[point.y] = 1;
                        is_col_visited_[col] = 0;
                    } else {
                        int count = 0;
                        points_[count] = point;

                        while (true) {
                            int row = FindInCol(points_[count].x, kStar);
                            if (row >= 0) {
                                count++;
                                points_[count] = cv::Point(points_[count - 1].x, row);
                                int col = FindInRow(points_[count].y, kPrime);
                                count++;
                                points_[count] = cv::Point(col, points_[count - 1].y);
                            } else {
                                break;
                            }
                        }

                        for (int i = 0; i < count + 1; i++) {
                            auto &mark = marked_.at<char>(points_[i].y, points_[i].x);
                            mark = mark == kStar ? 0 : kStar;
                        }

                        is_row_visited_ = std::vector<int>(n_, 0);
                        is_col_visited_ = std::vector<int>(n_, 0);

                        marked_.setTo(0, marked_ == kPrime);
                        break;
                    }
                }
            }
        }
    }

private:
    static constexpr int kStar = 1;
    static constexpr int kPrime = 2;

    cv::Mat dm_;
    cv::Mat marked_;
    std::vector<cv::Point> points_;

    std::vector<int> is_row_visited_;
    std::vector<int> is_col_visited_;

    int n_;
};

KuhnMunkres::KuhnMunkres() { impl_ = std::make_shared<Impl>(); }

std::vector<size_t> KuhnMunkres::Solve(const cv::Mat &dissimilarity_matrix) {
    CV_Assert(impl_ != nullptr);
    CV_Assert(!dissimilarity_matrix.empty());
    CV_Assert(dissimilarity_matrix.type() == CV_32F);

    return impl_->Solve(dissimilarity_matrix);
}

cv::Point Center(const cv::Rect &rect) {
    return cv::Point(static_cast<int>(rect.x + rect.width * 0.5),
                     static_cast<int>(rect.y + rect.height * 0.5));
}

TrackerParams::TrackerParams()
    : min_track_duration(25),
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

bool IsInRange(float x, const cv::Vec2f &v) { return v[0] <= x && x <= v[1]; }
bool IsInRange(float x, float a, float b) { return a <= x && x <= b; }

void Tracker::FilterDetectionsAndStore(const TrackedObjects &detections) {
    detections_.clear();
    for (const auto &det : detections) {
        float aspect_ratio = static_cast<float>(det.rect.height) / det.rect.width;
        if (det.confidence > params_.min_det_conf &&
                IsInRange(aspect_ratio, params_.bbox_aspect_ratios_range) &&
                IsInRange(static_cast<float>(det.rect.height), params_.bbox_heights_range)) {
            detections_.emplace_back(det);
        }
    }
}

void Tracker::SolveAssignmentProblem(
        const std::set<size_t> &track_ids, const TrackedObjects &detections,
        std::set<size_t> *unmatched_tracks, std::set<size_t> *unmatched_detections,
        std::set<std::tuple<size_t, size_t, float>> *matches) {
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
    for (auto id : track_ids) {
        if (res[i] < detections.size()) {
            matches->emplace(id, res[i], 1 - dissimilarity.at<float>(i, res[i]));
        } else {
            unmatched_tracks->insert(id);
        }
        i++;
    }
}

bool Tracker::EraseTrackIfBBoxIsOutOfFrame(size_t track_id) {
    if (tracks_.find(track_id) == tracks_.end()) return true;
    auto c = Center(tracks_.at(track_id).back().rect);
    if (frame_size_ != cv::Size() &&
            (c.x < 0 || c.y < 0 || c.x > frame_size_.width ||
             c.y > frame_size_.height)) {
        tracks_.at(track_id).lost = params_.forget_delay + 1;
        active_track_ids_.erase(track_id);
        return true;
    }
    return false;
}

bool Tracker::EraseTrackIfItWasLostTooManyFramesAgo(size_t track_id) {
    if (tracks_.find(track_id) == tracks_.end()) return true;
    if (tracks_.at(track_id).lost > params_.forget_delay) {
        active_track_ids_.erase(track_id);
        return true;
    }
    return false;
}

bool Tracker::UptateLostTrackAndEraseIfItsNeeded(size_t track_id) {
    tracks_.at(track_id).lost++;
    bool erased = EraseTrackIfBBoxIsOutOfFrame(track_id);
    if (!erased) erased = EraseTrackIfItWasLostTooManyFramesAgo(track_id);
    return erased;
}

void Tracker::UpdateLostTracks(const std::set<size_t> &track_ids) {
    for (auto track_id : track_ids) {
        UptateLostTrackAndEraseIfItsNeeded(track_id);
    }
}

void Tracker::Process(const cv::Mat &frame, const TrackedObjects &detections,
                      int frame_idx) {
    if (frame_size_ == cv::Size()) {
        frame_size_ = frame.size();
    } else {
        CV_Assert(frame_size_ == frame.size());
    }

    FilterDetectionsAndStore(detections);
    for (auto &obj : detections_) {
        obj.frame_idx = frame_idx;
    }

    auto active_tracks = active_track_ids_;

    if (!active_tracks.empty() && !detections_.empty()) {
        std::set<size_t> unmatched_tracks, unmatched_detections;
        std::set<std::tuple<size_t, size_t, float>> matches;

        SolveAssignmentProblem(active_tracks, detections_, &unmatched_tracks,
                               &unmatched_detections, &matches);

        for (const auto &match : matches) {
            size_t track_id = std::get<0>(match);
            size_t det_id = std::get<1>(match);
            float conf = std::get<2>(match);
            if (conf > params_.affinity_thr) {
                AppendToTrack(track_id, detections_[det_id]);
                unmatched_detections.erase(det_id);
            } else {
                unmatched_tracks.insert(track_id);
            }
        }

        AddNewTracks(detections_, unmatched_detections);
        UpdateLostTracks(unmatched_tracks);

        for (size_t id : active_tracks) {
            EraseTrackIfBBoxIsOutOfFrame(id);
        }
    } else {
        AddNewTracks(detections_);
        UpdateLostTracks(active_tracks);
    }

    if (params_.drop_forgotten_tracks) DropForgottenTracks();
}

void Tracker::DropForgottenTracks() {
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

void Tracker::DropForgottenTrack(size_t track_id) {
    CV_Assert(IsTrackForgotten(track_id));
    CV_Assert(active_track_ids_.count(track_id) == 0);
    tracks_.erase(track_id);
}

float Tracker::ShapeAffinity(const cv::Rect &trk, const cv::Rect &det) {
    float w_dist = static_cast<float>(std::fabs(trk.width - det.width)) / static_cast<float>(trk.width + det.width);
    float h_dist = static_cast<float>(std::fabs(trk.height - det.height)) / static_cast<float>(trk.height + det.height);
    return exp(-params_.shape_affinity_w * (w_dist + h_dist));
}

float Tracker::MotionAffinity(const cv::Rect &trk, const cv::Rect &det) {
    float x_dist = static_cast<float>(trk.x - det.x) * (trk.x - det.x) /
            (det.width * det.width);
    float y_dist = static_cast<float>(trk.y - det.y) * (trk.y - det.y) /
            (det.height * det.height);
    return exp(-params_.motion_affinity_w * (x_dist + y_dist));
}

void Tracker::ComputeDissimilarityMatrix(const std::set<size_t> &active_tracks,
                                         const TrackedObjects &detections,
                                         cv::Mat *dissimilarity_matrix) {
    dissimilarity_matrix->create(active_tracks.size(), detections.size(), CV_32F);
    size_t i = 0;
    for (auto id : active_tracks) {
        auto ptr = dissimilarity_matrix->ptr<float>(i);
        for (size_t j = 0; j < detections.size(); j++) {
            auto last_det = tracks_.at(id).objects.back();
            ptr[j] = Distance(last_det, detections[j]);
        }
        i++;
    }
}

void Tracker::AddNewTracks(const TrackedObjects &detections) {
    for (size_t i = 0; i < detections.size(); i++) {
        AddNewTrack(detections[i]);
    }
}

void Tracker::AddNewTracks(const TrackedObjects &detections,
                           const std::set<size_t> &ids) {
    for (size_t i : ids) {
        CV_Assert(i < detections.size());
        AddNewTrack(detections[i]);
    }
}

void Tracker::AddNewTrack(const TrackedObject &detection) {
    auto detection_with_id = detection;
    detection_with_id.object_id = tracks_counter_;
    tracks_.emplace(
                std::pair<size_t, Track>(tracks_counter_, Track({detection_with_id})));

    active_track_ids_.insert(tracks_counter_);
    tracks_counter_++;
}

void Tracker::AppendToTrack(size_t track_id, const TrackedObject &detection) {
    CV_Assert(!IsTrackForgotten(track_id));

    auto detection_with_id = detection;
    detection_with_id.object_id = track_id;

    auto &track = tracks_.at(track_id);

    track.objects.emplace_back(detection_with_id);
    track.lost = 0;
    track.length++;

    if (params_.max_num_objects_in_track > 0) {
        while (track.size() >
               static_cast<size_t>(params_.max_num_objects_in_track)) {
            track.objects.erase(track.objects.begin());
        }
    }
}

float Tracker::Distance(const TrackedObject &obj1, const TrackedObject &obj2) {
    const float eps = 1e-6f;
    float shp_aff = ShapeAffinity(obj1.rect, obj2.rect);
    if (shp_aff < eps) return 1.0;

    float mot_aff = MotionAffinity(obj1.rect, obj2.rect);
    if (mot_aff < eps) return 1.0;

    return 1.0f - shp_aff * mot_aff;
}

bool Tracker::IsTrackValid(size_t id) const {
    const auto &track = tracks_.at(id);
    const auto &objects = track.objects;
    if (objects.empty()) {
        return false;
    }
    size_t duration_frames = objects.back().frame_idx - track.first_object.frame_idx;
    if (duration_frames < params_.min_track_duration)
        return false;
    return true;
}

bool Tracker::IsTrackForgotten(size_t id) const {
    return tracks_.at(id).lost > params_.forget_delay;
}

void Tracker::Reset() {
    active_track_ids_.clear();
    tracks_.clear();

    detections_.clear();

    tracks_counter_ = 0;
    valid_tracks_counter_ = 0;

    frame_size_ = cv::Size();
}

size_t Tracker::Count() const {
    size_t count = valid_tracks_counter_;
    for (const auto &pair : tracks_) {
        count += (IsTrackValid(pair.first) ? 1 : 0);
    }
    return count;
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
            size_t start = static_cast<int>(track.objects.size()) >= params_.averaging_window_size_for_rects ?
                        track.objects.size() - params_.averaging_window_size_for_rects : 0;

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

            object.label = LabelWithMaxFrequencyInTrack(track, params_.averaging_window_size_for_labels);
            object.object_id = idx;

            detections.push_back(object);
        }
    }
    return detections;
}

int LabelWithMaxFrequencyInTrack(const Track &track, int window_size) {
    std::unordered_map<int, int> frequencies;
    int max_frequent_count = 0;
    int max_frequent_id = TrackedObject::UNKNOWN_LABEL_IDX;

    int start = static_cast<int>(track.objects.size()) >= window_size ?
        static_cast<int>(track.objects.size()) - window_size : 0;

    for (size_t i = start; i < track.objects.size(); i++) {
        const auto & detection = track.objects[i];
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

const std::unordered_map<size_t, Track> &Tracker::tracks() const {
    return tracks_;
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
