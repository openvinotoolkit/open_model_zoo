// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <chrono>  // NOLINT

#include <algorithm>
#include <deque>
#include <map>
#include <memory>
#include <limits>
#include <set>
#include <string>
#include <vector>
#include <utility>

#include "openvino/openvino.hpp"

#include "gflags/gflags.h"
#include "monitors/presenter.h"
#include "utils/args_helper.hpp"
#include "utils/images_capture.h"
#include "utils/ocv_common.hpp"
#include "utils/slog.hpp"
#include "cnn.hpp"
#include "actions.hpp"
#include "action_detector.hpp"
#include "detector.hpp"
#include "face_reid.hpp"
#include "tracker.hpp"
#include "logger.hpp"
#include "smart_classroom_demo.hpp"

namespace {

class Visualizer {
private:
    cv::Mat m_frame;
    cv::Mat m_top_persons;
    const bool m_enabled;
    const int m_num_top_persons;
    LazyVideoWriter m_writer;
    float m_rect_scale_x;
    float m_rect_scale_y;
    static int const m_max_input_width = 1920;
    std::string const m_main_window_name = "Smart classroom demo";
    std::string const m_top_window_name = "Top-k students";
    static int const m_crop_width = 128;
    static int const m_crop_height = 320;
    static int const m_header_size = 80;
    static int const m_margin_size = 5;

public:
    Visualizer(bool enabled, const LazyVideoWriter& writer, int num_top_persons) :
        m_enabled(enabled), m_num_top_persons(num_top_persons), m_writer(writer), m_rect_scale_x(0), m_rect_scale_y(0) {
        if (!m_enabled) {
            return;
        }

        cv::namedWindow(m_main_window_name);

        if (m_num_top_persons > 0) {
            cv::namedWindow(m_top_window_name);

            CreateTopWindow();
            ClearTopWindow();
        }
    }

    static cv::Size GetOutputSize(const cv::Size& input_size) {
        if (input_size.width > m_max_input_width) {
            float ratio = static_cast<float>(input_size.height) / input_size.width;
            return cv::Size(m_max_input_width, cvRound(ratio*m_max_input_width));
        }
        return input_size;
    }

    void SetFrame(const cv::Mat& frame) {
        if (!m_enabled) {
            return;
        }

        m_frame = frame.clone();
        m_rect_scale_x = 1;
        m_rect_scale_y = 1;
        cv::Size new_size = GetOutputSize(m_frame.size());
        if (new_size != m_frame.size()) {
            m_rect_scale_x = static_cast<float>(new_size.height) / m_frame.size().height;
            m_rect_scale_y = static_cast<float>(new_size.width) / m_frame.size().width;
            cv::resize(m_frame, m_frame, new_size);
        }
    }

    void Show() {
        if (m_enabled) {
            cv::imshow(m_main_window_name, m_frame);
        }

        m_writer.write(m_frame);
    }

    void DrawCrop(cv::Rect roi, int id, const cv::Scalar& color) const {
        if (!m_enabled || m_num_top_persons <= 0) {
            return;
        }

        if (id < 0 || id >= m_num_top_persons) {
            return;
        }

        if (m_rect_scale_x != 1 || m_rect_scale_y != 1) {
            roi.x = cvRound(roi.x * m_rect_scale_x);
            roi.y = cvRound(roi.y * m_rect_scale_y);

            roi.height = cvRound(roi.height * m_rect_scale_y);
            roi.width = cvRound(roi.width * m_rect_scale_x);
        }

        roi.x = std::max(0, roi.x);
        roi.y = std::max(0, roi.y);
        roi.width = std::min(roi.width, m_frame.cols - roi.x);
        roi.height = std::min(roi.height, m_frame.rows - roi.y);

        const auto crop_label = std::to_string(id + 1);

        auto frame_crop = m_frame(roi).clone();
        cv::resize(frame_crop, frame_crop, cv::Size(m_crop_width, m_crop_height));

        const int shift = (id + 1) * m_margin_size + id * m_crop_width;
        frame_crop.copyTo(m_top_persons(cv::Rect(shift, m_header_size, m_crop_width, m_crop_height)));

        cv::imshow(m_top_window_name, m_top_persons);
    }

    void DrawObject(cv::Rect rect, const std::string& label_to_draw,
                    const cv::Scalar& text_color, const cv::Scalar& bbox_color, bool plot_bg) {
        if (!m_enabled) {
            return;
        }

        if (m_rect_scale_x != 1 || m_rect_scale_y != 1) {
            rect.x = cvRound(rect.x * m_rect_scale_x);
            rect.y = cvRound(rect.y * m_rect_scale_y);

            rect.height = cvRound(rect.height * m_rect_scale_y);
            rect.width = cvRound(rect.width * m_rect_scale_x);
        }
        cv::rectangle(m_frame, rect, bbox_color);

        if (plot_bg && !label_to_draw.empty()) {
            int baseLine = 0;
            const cv::Size label_size = cv::getTextSize(label_to_draw, cv::FONT_HERSHEY_PLAIN, 1, 1, &baseLine);
            cv::rectangle(
                m_frame,
                cv::Point(rect.x, rect.y - label_size.height),
                cv::Point(rect.x + label_size.width, rect.y + baseLine),
                bbox_color, cv::FILLED);
        }
        if (!label_to_draw.empty()) {
            cv::putText(m_frame, label_to_draw, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_PLAIN, 1,
                        text_color, 1, cv::LINE_AA);
        }
    }

    void CreateTopWindow() {
        if (!m_enabled || m_num_top_persons <= 0) {
            return;
        }

        const int width = m_margin_size * (m_num_top_persons + 1) + m_crop_width * m_num_top_persons;
        const int height = m_header_size + m_crop_height + m_margin_size;

        m_top_persons.create(height, width, CV_8UC3);
    }

    void ClearTopWindow() {
        if (!m_enabled || m_num_top_persons <= 0) {
            return;
        }

        m_top_persons.setTo(cv::Scalar(255, 255, 255));

        for (int i = 0; i < m_num_top_persons; ++i) {
            const int shift = (i + 1) * m_margin_size + i * m_crop_width;

            cv::rectangle(m_top_persons, cv::Point(shift, m_header_size),
                          cv::Point(shift + m_crop_width, m_header_size + m_crop_height),
                          cv::Scalar(128, 128, 128), cv::FILLED);

            const auto label_to_draw = "#" + std::to_string(i + 1);
            int baseLine = 0;
            const auto label_size = cv::getTextSize(label_to_draw, cv::FONT_HERSHEY_SIMPLEX, 2, 2, &baseLine);
            const int text_shift = (m_crop_width - label_size.width) / 2;
            cv::putText(m_top_persons, label_to_draw,
                        cv::Point(shift + text_shift, label_size.height + baseLine / 2),
                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        }

        cv::imshow(m_top_window_name, m_top_persons);
    }

    void Finalize() const {
        if (m_enabled) {
            cv::destroyWindow(m_main_window_name);

            if (m_num_top_persons > 0) {
                cv::destroyWindow(m_top_window_name);
            }
        }
    }
};

const int default_action_index = -1;  // Unknown action class

void ConvertActionMapsToFrameEventTracks(const std::vector<std::map<int, int>>& obj_id_to_action_maps,
                                         int default_action,
                                         std::map<int, FrameEventsTrack>* obj_id_to_actions_track) {
    for (size_t frame_id = 0; frame_id < obj_id_to_action_maps.size(); ++frame_id) {
        for (const auto& tup : obj_id_to_action_maps[frame_id]) {
            if (tup.second != default_action) {
                (*obj_id_to_actions_track)[tup.first].emplace_back(frame_id, tup.second);
            }
        }
    }
}

void SmoothTracks(const std::map<int, FrameEventsTrack>& obj_id_to_actions_track,
                  int start_frame, int end_frame, int window_size, int min_length, int default_action,
                  std::map<int, RangeEventsTrack>* obj_id_to_events) {
    // Iterate over face tracks
    for (const auto& tup : obj_id_to_actions_track) {
        const auto& frame_events = tup.second;
        if (frame_events.empty()) {
            continue;
        }

        RangeEventsTrack range_events;


        // Merge neighbouring events and filter short ones
        range_events.emplace_back(frame_events.front().frame_id,
                                  frame_events.front().frame_id + 1,
                                  frame_events.front().action);

        for (size_t frame_id = 1; frame_id < frame_events.size(); ++frame_id) {
            const auto& last_range_event = range_events.back();
            const auto& cur_frame_event = frame_events[frame_id];

            if (last_range_event.end_frame_id + window_size - 1 >= cur_frame_event.frame_id &&
                last_range_event.action == cur_frame_event.action) {
                range_events.back().end_frame_id = cur_frame_event.frame_id + 1;
            } else {
                if (range_events.back().end_frame_id - range_events.back().begin_frame_id < min_length) {
                    range_events.pop_back();
                }

                range_events.emplace_back(cur_frame_event.frame_id,
                                          cur_frame_event.frame_id + 1,
                                          cur_frame_event.action);
            }
        }
        if (range_events.back().end_frame_id - range_events.back().begin_frame_id < min_length) {
            range_events.pop_back();
        }

        // Extrapolate track
        if (range_events.empty()) {
            range_events.emplace_back(start_frame, end_frame, default_action);
        } else {
            range_events.front().begin_frame_id = start_frame;
            range_events.back().end_frame_id = end_frame;
        }

        // Interpolate track
        for (size_t event_id = 1; event_id < range_events.size(); ++event_id) {
            auto& last_event = range_events[event_id - 1];
            auto& cur_event = range_events[event_id];

            int middle_point = static_cast<int>(0.5f * (cur_event.begin_frame_id + last_event.end_frame_id));

            cur_event.begin_frame_id = middle_point;
            last_event.end_frame_id = middle_point;
        }

        // Merge consecutive events
        auto& final_events = (*obj_id_to_events)[tup.first];
        final_events.push_back(range_events.front());
        for (size_t event_id = 1; event_id < range_events.size(); ++event_id) {
            const auto& cur_event = range_events[event_id];

            if (final_events.back().action == cur_event.action) {
                final_events.back().end_frame_id = cur_event.end_frame_id;
            } else {
                final_events.push_back(cur_event);
            }
        }
    }
}

void ConvertRangeEventsTracksToActionMaps(int num_frames,
                                          const std::map<int, RangeEventsTrack>& obj_id_to_events,
                                          std::vector<std::map<int, int>>* obj_id_to_action_maps) {
    obj_id_to_action_maps->resize(num_frames);

    for (const auto& tup : obj_id_to_events) {
        const int obj_id = tup.first;
        const auto& events = tup.second;

        for (const auto& event : events) {
            for (int frame_id = event.begin_frame_id; frame_id < event.end_frame_id; ++frame_id) {
                (*obj_id_to_action_maps)[frame_id].emplace(obj_id, event.action);
            }
        }
    }
}

std::string GetActionTextLabel(const unsigned label, const std::vector<std::string>& actions_map) {
    if (label < actions_map.size()) {
        return actions_map[label];
    }
    return "__undefined__";
}

cv::Scalar GetActionTextColor(const unsigned label) {
    static const cv::Scalar label_colors[] = {
        cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0), cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 255)};
    if (label < arraySize(label_colors)) {
        return label_colors[label];
    }
    return cv::Scalar(0, 0, 0);
}

float CalculateIoM(const cv::Rect& rect1, const cv::Rect& rect2) {
  int area1 = rect1.area();
  int area2 = rect2.area();

  float area_min = static_cast<float>(std::min(area1, area2));
  float area_intersect = static_cast<float>((rect1 & rect2).area());

  return area_intersect / area_min;
}

cv::Rect DecreaseRectByRelBorders(const cv::Rect& r) {
    float w = static_cast<float>(r.width);
    float h = static_cast<float>(r.height);

    float left = std::ceil(w * 0.0f);
    float top = std::ceil(h * 0.0f);
    float right = std::ceil(w * 0.0f);
    float bottom = std::ceil(h * .7f);

    cv::Rect res;
    res.x = r.x + static_cast<int>(left);
    res.y = r.y + static_cast<int>(top);
    res.width = static_cast<int>(r.width - left - right);
    res.height = static_cast<int>(r.height - top - bottom);
    return res;
}

int GetIndexOfTheNearestPerson(const TrackedObject& face, const std::vector<TrackedObject>& tracked_persons) {
    int argmax = -1;
    float max_iom = std::numeric_limits<float>::lowest();
    for (size_t i = 0; i < tracked_persons.size(); i++) {
        float iom = CalculateIoM(face.rect, DecreaseRectByRelBorders(tracked_persons[i].rect));
        if ((iom > 0) && (iom > max_iom)) {
            max_iom = iom;
            argmax = i;
        }
    }
    return argmax;
}

std::map<int, int> GetMapFaceTrackIdToLabel(const std::vector<Track>& face_tracks) {
    std::map<int, int> face_track_id_to_label;
    for (const auto& track : face_tracks) {
        const auto& first_obj = track.first_object;
        // check consistency
        // to receive this consistency for labels
        // use the function UpdateTrackLabelsToBestAndFilterOutUnknowns
        for (const auto& obj : track.objects) {
            SCR_CHECK_EQ(obj.label, first_obj.label);
            SCR_CHECK_EQ(obj.object_id, first_obj.object_id);
        }

        auto cur_obj_id = first_obj.object_id;
        auto cur_label = first_obj.label;
        if (face_track_id_to_label.count(cur_obj_id) == 1) {
            throw std::runtime_error("Repeating face tracks");
        }
        face_track_id_to_label[cur_obj_id] = cur_label;
    }
    return face_track_id_to_label;
}

class FaceRecognizer {
public:
    virtual ~FaceRecognizer() = default;

    virtual bool LabelExists(const std::string& label) const = 0;
    virtual std::string GetLabelByID(int id) const = 0;
    virtual std::vector<std::string> GetIDToLabelMap() const = 0;

    virtual std::vector<int> Recognize(const cv::Mat& frame, const detection::DetectedObjects& faces) = 0;
};

class FaceRecognizerNull : public FaceRecognizer {
public:
    bool LabelExists(const std::string&) const override { return false; }

    std::string GetLabelByID(int) const override {
        return EmbeddingsGallery::unknown_label;
    }

    std::vector<std::string> GetIDToLabelMap() const override { return {}; }

    std::vector<int> Recognize(const cv::Mat&, const detection::DetectedObjects& faces) override {
        return std::vector<int>(faces.size(), EmbeddingsGallery::unknown_id);
    }
};

class FaceRecognizerDefault : public FaceRecognizer {
public:
    FaceRecognizerDefault(
            const CnnConfig& landmarks_detector_config,
            const CnnConfig& reid_config,
            const detection::DetectorConfig& face_registration_det_config,
            const std::string& face_gallery_path,
            double reid_threshold,
            int min_size_fr,
            bool crop_gallery,
            bool greedy_reid_matching) :
        landmarks_detector(landmarks_detector_config),
        face_reid(reid_config),
        face_gallery(face_gallery_path, reid_threshold, min_size_fr, crop_gallery,
                     face_registration_det_config, landmarks_detector, face_reid,
                     greedy_reid_matching)
    {
        if (face_gallery.size() == 0) {
            slog::warn << "Face reid gallery is empty!" << slog::endl;
        } else {
            slog::info << "Face reid gallery size: " << face_gallery.size() << slog::endl;
        }
    }

    bool LabelExists(const std::string& label) const override {
        return face_gallery.LabelExists(label);
    }

    std::string GetLabelByID(int id) const override {
        return face_gallery.GetLabelByID(id);
    }

    std::vector<std::string> GetIDToLabelMap() const override {
        return face_gallery.GetIDToLabelMap();
    }

    std::vector<int> Recognize(const cv::Mat& frame, const detection::DetectedObjects& faces) override {
        const int maxLandmarksBatch = landmarks_detector.maxBatchSize();
        int numFaces = (int)faces.size();

        std::vector<cv::Mat> landmarks;
        std::vector<cv::Mat> embeddings;
        std::vector<cv::Mat> face_rois;

        auto face_roi = [&](const detection::DetectedObject& face) {
            return frame(face.rect);
        };
        if (numFaces < maxLandmarksBatch) {
            std::transform(faces.begin(), faces.end(), std::back_inserter(face_rois), face_roi);
            landmarks_detector.Compute(face_rois, &landmarks, cv::Size(2, 5));
            AlignFaces(&face_rois, &landmarks);
            face_reid.Compute(face_rois, &embeddings);
        } else {
            auto embedding = [&](cv::Mat& emb) { return emb; };
            for (int n = numFaces; n > 0; n -= maxLandmarksBatch) {
                landmarks.clear();
                face_rois.clear();
                size_t start_idx = size_t(numFaces) - n;
                size_t end_idx = start_idx + std::min(numFaces, maxLandmarksBatch);
                std::transform(faces.begin() + start_idx, faces.begin() + end_idx, std::back_inserter(face_rois), face_roi);

                landmarks_detector.Compute(face_rois, &landmarks, cv::Size(2, 5));

                AlignFaces(&face_rois, &landmarks);

                std::vector<cv::Mat> batch_embeddings;
                face_reid.Compute(face_rois, &batch_embeddings);
                std::transform(batch_embeddings.begin(), batch_embeddings.end(), std::back_inserter(embeddings), embedding);
            }
        }

        return face_gallery.GetIDsByEmbeddings(embeddings);
    }

private:
    VectorCNN landmarks_detector;
    VectorCNN face_reid;
    EmbeddingsGallery face_gallery;
};

bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    // Parsing and validation of input args

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }
    if (FLAGS_m_act.empty() && FLAGS_m_fd.empty()) {
        throw std::logic_error("At least one parameter -m_act or -m_fd must be set");
    }

    return true;
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        PerformanceMetrics metrics;

        // This demo covers 4 certain topologies and cannot be generalized
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        const auto ad_model_path = FLAGS_m_act;
        const auto fd_model_path = FLAGS_m_fd;
        const auto fr_model_path = FLAGS_m_reid;
        const auto lm_model_path = FLAGS_m_lm;
        const auto teacher_id = FLAGS_teacher_id;

        if (!FLAGS_teacher_id.empty() && !FLAGS_top_id.empty()) {
            slog::err << "Cannot run simultaneously teacher action and top-k students recognition."
                      << slog::endl;
            return 1;
        }

        const auto actions_type = FLAGS_teacher_id.empty() ?
            (FLAGS_a_top > 0 ? TOP_K : STUDENT) :
            TEACHER;
        if (!FLAGS_ad.empty()) {
            if (actions_type != STUDENT) {
                slog::err << "-ad requires -teacher_id and -a_top to be unset" << slog::endl;
                return 1;
            }
            if (FLAGS_fg.empty()) {
                slog::err << "-ad requires -fg to be set" << slog::endl;
                return 1;
            }
        }
        const auto actions_map = actions_type == STUDENT ?
            split(FLAGS_student_ac, ',') : actions_type == TOP_K ?
            split(FLAGS_top_ac, ',') :
            split(FLAGS_teacher_ac, ',');
        const auto num_top_persons = actions_type == TOP_K ? FLAGS_a_top : -1;
        const auto top_action_id = actions_type == TOP_K ?
            std::distance(actions_map.begin(), find(actions_map.begin(), actions_map.end(), FLAGS_top_id)) : -1;
        if (actions_type == TOP_K && (top_action_id < 0 || top_action_id >= static_cast<int>(actions_map.size()))) {
            slog::err << "Cannot find target action: " << FLAGS_top_id << slog::endl;
            return 1;
        }

        slog::info << ov::get_openvino_version() << slog::endl;
        ov::Core core;

        std::unique_ptr<AsyncDetection<DetectedAction>> action_detector;
        if (!ad_model_path.empty()) {
            // Load action detector
            ActionDetectorConfig action_config(ad_model_path, "Person/Action Detection");
            action_config.m_deviceName = FLAGS_d_act;
            action_config.m_core = core;
            action_config.is_async = true;
            action_config.detection_confidence_threshold = static_cast<float>(FLAGS_t_ad);
            action_config.action_confidence_threshold = static_cast<float>(FLAGS_t_ar);
            action_config.num_action_classes = actions_map.size();
            action_detector.reset(new ActionDetection(action_config));
        } else {
            action_detector.reset(new NullDetection<DetectedAction>);
            slog::info << "Person/Action Detection DISABLED" << slog::endl;
        }

        std::unique_ptr<AsyncDetection<detection::DetectedObject>> face_detector;
        if (!fd_model_path.empty()) {
            // Load face detector
            detection::DetectorConfig face_config(fd_model_path);
            face_config.m_deviceName = FLAGS_d_fd;
            face_config.m_core = core;
            face_config.is_async = true;
            face_config.confidence_threshold = static_cast<float>(FLAGS_t_fd);
            face_config.input_h = FLAGS_inh_fd;
            face_config.input_w = FLAGS_inw_fd;
            face_config.increase_scale_x = static_cast<float>(FLAGS_exp_r_fd);
            face_config.increase_scale_y = static_cast<float>(FLAGS_exp_r_fd);
            face_detector.reset(new detection::FaceDetection(face_config));
        } else {
            face_detector.reset(new NullDetection<detection::DetectedObject>);
        }

        std::unique_ptr<FaceRecognizer> face_recognizer;

        if (!fd_model_path.empty() && !fr_model_path.empty() && !lm_model_path.empty()) {
            // Create face recognizer
            detection::DetectorConfig face_registration_det_config(fd_model_path);
            face_registration_det_config.m_deviceName = FLAGS_d_fd;
            face_registration_det_config.m_core = core;
            face_registration_det_config.is_async = false;
            face_registration_det_config.confidence_threshold = static_cast<float>(FLAGS_t_reg_fd);
            face_registration_det_config.increase_scale_x = static_cast<float>(FLAGS_exp_r_fd);
            face_registration_det_config.increase_scale_y = static_cast<float>(FLAGS_exp_r_fd);

            CnnConfig reid_config(fr_model_path, "Face Re-Identification");
            reid_config.m_deviceName = FLAGS_d_reid;
            reid_config.m_max_batch_size = 16;
            reid_config.m_core = core;

            CnnConfig landmarks_config(lm_model_path, "Facial Landmarks Regression");
            landmarks_config.m_deviceName = FLAGS_d_lm;
            landmarks_config.m_max_batch_size = 16;
            landmarks_config.m_core = core;
            face_recognizer.reset(new FaceRecognizerDefault(
                landmarks_config, reid_config,
                face_registration_det_config,
                FLAGS_fg, FLAGS_t_reid, FLAGS_min_size_fr, FLAGS_crop_gallery, FLAGS_greedy_reid_matching));

            if (actions_type == TEACHER && !face_recognizer->LabelExists(teacher_id)) {
                slog::err << "Teacher id does not exist in the gallery!" << slog::endl;
                return 1;
            }
        } else {
            slog::warn << "Face Recognition models are disabled!" << slog::endl;
            if (actions_type == TEACHER) {
                slog::err << "Face recognition must be enabled to recognize teacher actions." << slog::endl;
                return 1;
            }

            face_recognizer.reset(new FaceRecognizerNull);
        }

        // Create tracker for reid
        TrackerParams tracker_reid_params;
        tracker_reid_params.min_track_duration = 1;
        tracker_reid_params.forget_delay = 150;
        tracker_reid_params.affinity_thr = 0.8f;
        tracker_reid_params.averaging_window_size_for_rects = 1;
        tracker_reid_params.averaging_window_size_for_labels = std::numeric_limits<int>::max();
        tracker_reid_params.bbox_heights_range = cv::Vec2f(10, 1080);
        tracker_reid_params.drop_forgotten_tracks = false;
        tracker_reid_params.max_num_objects_in_track = std::numeric_limits<int>::max();
        tracker_reid_params.objects_type = "face";

        Tracker tracker_reid(tracker_reid_params);

        // Create Tracker for action recognition
        TrackerParams tracker_action_params;
        tracker_action_params.min_track_duration = 8;
        tracker_action_params.forget_delay = 150;
        tracker_action_params.affinity_thr = 0.9f;
        tracker_action_params.averaging_window_size_for_rects = 5;
        tracker_action_params.averaging_window_size_for_labels = FLAGS_ss_t > 0 ?
            FLAGS_ss_t : actions_type == TOP_K ? 5 : 1;
        tracker_action_params.bbox_heights_range = cv::Vec2f(10, 2160);
        tracker_action_params.drop_forgotten_tracks = false;
        tracker_action_params.max_num_objects_in_track = std::numeric_limits<int>::max();
        tracker_action_params.objects_type = "action";

        Tracker tracker_action(tracker_action_params);

        size_t work_num_frames = 0;
        size_t wait_num_frames = 0;
        size_t total_num_frames = 0;
        const char ESC_KEY = 27;
        const char SPACE_KEY = 32;
        const cv::Scalar green_color(0, 255, 0);
        const cv::Scalar red_color(0, 0, 255);
        const cv::Scalar white_color(255, 255, 255);
        std::vector<std::map<int, int>> face_obj_id_to_action_maps;
        std::map<int, int> top_k_obj_ids;

        int teacher_track_id = -1;

        std::unique_ptr<ImagesCapture> cap = openImagesCapture(FLAGS_i, FLAGS_loop, read_type::safe, 0, FLAGS_read_limit);
        cv::Mat frame = cap->read();

        cv::Size graphSize{static_cast<int>(frame.cols / 4), 60};
        Presenter presenter(FLAGS_u, frame.rows - graphSize.height - 10, graphSize);

        LazyVideoWriter videoWriter{FLAGS_o, cap->fps(), FLAGS_limit};
        Visualizer sc_visualizer(!FLAGS_no_show, videoWriter, num_top_persons);
        DetectionsLogger logger(slog::debug, FLAGS_r, FLAGS_ad, FLAGS_al);
        if (actions_type != TOP_K) {
            action_detector->enqueue(frame);
            action_detector->submitRequest();
            face_detector->enqueue(frame);
            face_detector->submitRequest();
        }

        bool is_monitoring_enabled = false;

        bool is_last_frame = false;
        while (!is_last_frame) {
            auto startTime = std::chrono::steady_clock::now();
            cv::Mat prev_frame = std::move(frame);
            frame = cap->read();
            if (frame.data && frame.size() != prev_frame.size()) {
                throw std::runtime_error("Can't track objects on images of different size");
            }
            is_last_frame = !frame.data;

            logger.CreateNextFrameRecord(FLAGS_i, work_num_frames, prev_frame.cols, prev_frame.rows);
            char key = cv::waitKey(1);
            if (key == ESC_KEY) {
                break;
            }
            presenter.handleKey(key);

            presenter.drawGraphs(prev_frame);

            sc_visualizer.SetFrame(prev_frame);
            if (actions_type == TOP_K) {
                if ( (is_monitoring_enabled && key == SPACE_KEY) || (!is_monitoring_enabled && key != SPACE_KEY) ) {
                    if (key == SPACE_KEY) {
                        action_detector->wait();
                        action_detector->fetchResults();

                        tracker_action.Reset();
                        top_k_obj_ids.clear();

                        is_monitoring_enabled = false;

                        sc_visualizer.ClearTopWindow();
                    }

                    ++wait_num_frames;
                    metrics.update(startTime, frame, { 10, 22 }, cv::FONT_HERSHEY_COMPLEX, 0.65);
                } else {
                    if (key == SPACE_KEY) {
                        is_monitoring_enabled = true;

                        action_detector->enqueue(prev_frame);
                        action_detector->submitRequest();
                    }

                    action_detector->wait();
                    DetectedActions actions = action_detector->fetchResults();

                    if (!is_last_frame) {
                        action_detector->enqueue(frame);
                        action_detector->submitRequest();
                    }

                    TrackedObjects tracked_action_objects;
                    for (const auto& action : actions) {
                        tracked_action_objects.emplace_back(action.rect, action.detection_conf, action.label);
                    }

                    tracker_action.Process(prev_frame, tracked_action_objects, total_num_frames);
                    const auto tracked_actions = tracker_action.TrackedDetectionsWithLabels();

                    if (static_cast<int>(top_k_obj_ids.size()) < FLAGS_a_top) {
                        for (const auto& action : tracked_actions) {
                            if (action.label == top_action_id && top_k_obj_ids.count(action.object_id) == 0) {
                                const int action_id_in_top = top_k_obj_ids.size();
                                top_k_obj_ids.emplace(action.object_id, action_id_in_top);

                                sc_visualizer.DrawCrop(action.rect, action_id_in_top, red_color);

                                if (static_cast<int>(top_k_obj_ids.size()) >= FLAGS_a_top) {
                                    break;
                                }
                            }
                        }
                    }

                    for (const auto& action : tracked_actions) {
                        auto box_color = white_color;
                        std::string box_caption = "";

                        if (top_k_obj_ids.count(action.object_id) > 0) {
                            box_color = red_color;
                            box_caption = std::to_string(top_k_obj_ids[action.object_id] + 1);
                        }

                        sc_visualizer.DrawObject(action.rect, box_caption, white_color, box_color, true);
                    }
                    ++work_num_frames;
                    metrics.update(startTime, frame, { 10, 22 }, cv::FONT_HERSHEY_COMPLEX, 0.65);
                }
            } else {
                face_detector->wait();
                detection::DetectedObjects faces = face_detector->fetchResults();
                action_detector->wait();
                DetectedActions actions = action_detector->fetchResults();
                if (!is_last_frame) {
                    face_detector->enqueue(frame);
                    face_detector->submitRequest();
                    action_detector->enqueue(frame);
                    action_detector->submitRequest();
                }
                auto ids = face_recognizer->Recognize(prev_frame, faces);
                TrackedObjects tracked_face_objects;

                for (size_t i = 0; i < faces.size(); i++) {
                    tracked_face_objects.emplace_back(faces[i].rect, faces[i].confidence, ids[i]);
                }
                tracker_reid.Process(prev_frame, tracked_face_objects, work_num_frames);
                const auto tracked_faces = tracker_reid.TrackedDetectionsWithLabels();

                TrackedObjects tracked_action_objects;
                for (const auto& action : actions) {
                    tracked_action_objects.emplace_back(action.rect, action.detection_conf, action.label);
                }
                tracker_action.Process(prev_frame, tracked_action_objects, work_num_frames);
                const auto tracked_actions = tracker_action.TrackedDetectionsWithLabels();

                std::map<int, int> frame_face_obj_id_to_action;
                for (size_t j = 0; j < tracked_faces.size(); j++) {
                    const auto& face = tracked_faces[j];
                    std::string face_label = face_recognizer->GetLabelByID(face.label);

                    std::string label_to_draw;
                    if (face.label != EmbeddingsGallery::unknown_id)
                        label_to_draw += face_label;

                    int person_ind = GetIndexOfTheNearestPerson(face, tracked_actions);
                    int action_ind = default_action_index;
                    if (person_ind >= 0) {
                        action_ind = tracked_actions[person_ind].label;
                    }

                    if (actions_type == STUDENT) {
                        if (action_ind != default_action_index) {
                            label_to_draw += "[" + GetActionTextLabel(action_ind, actions_map) + "]";
                        }
                        frame_face_obj_id_to_action[face.object_id] = action_ind;
                        sc_visualizer.DrawObject(face.rect, label_to_draw, red_color, white_color, true);
                        logger.AddFaceToFrame(face.rect, face_label, "");
                    }

                    if ((actions_type == TEACHER) && (person_ind >= 0)) {
                        if (face_label == teacher_id) {
                            teacher_track_id = tracked_actions[person_ind].object_id;
                        } else if (teacher_track_id == tracked_actions[person_ind].object_id) {
                            teacher_track_id = -1;
                        }
                    }
                }

                if (actions_type == STUDENT) {
                    for (const auto& action : tracked_actions) {
                        const auto& action_label = GetActionTextLabel(action.label, actions_map);
                        const auto& action_color = GetActionTextColor(action.label);
                        const auto& text_label = fd_model_path.empty() ? action_label : "";
                        sc_visualizer.DrawObject(action.rect, text_label, action_color, white_color, true);
                        logger.AddPersonToFrame(action.rect, action_label, "");
                        logger.AddDetectionToFrame(action, work_num_frames);
                    }
                    face_obj_id_to_action_maps.push_back(frame_face_obj_id_to_action);
                } else if (teacher_track_id >= 0) {
                    auto res_find = std::find_if(tracked_actions.begin(), tracked_actions.end(),
                                [teacher_track_id](const TrackedObject& o){ return o.object_id == teacher_track_id; });
                    if (res_find != tracked_actions.end()) {
                        const auto& track_action = *res_find;
                        const auto& action_label = GetActionTextLabel(track_action.label, actions_map);
                        sc_visualizer.DrawObject(track_action.rect, action_label, red_color, white_color, true);
                        logger.AddPersonToFrame(track_action.rect, action_label, teacher_id);
                    }
                }
                metrics.update(startTime, frame, { 10, 22 }, cv::FONT_HERSHEY_COMPLEX, 0.65);
                ++work_num_frames;
            }

            ++total_num_frames;

            sc_visualizer.Show();

            logger.FinalizeFrameRecord();
        }
        sc_visualizer.Finalize();

        if (actions_type == STUDENT) {
            auto face_tracks = tracker_reid.vector_tracks();

            // correct labels for track
            std::vector<Track> new_face_tracks = UpdateTrackLabelsToBestAndFilterOutUnknowns(face_tracks);
            std::map<int, int> face_track_id_to_label = GetMapFaceTrackIdToLabel(new_face_tracks);

            std::vector<std::string> face_id_to_label_map = face_recognizer->GetIDToLabelMap();

            if (!face_id_to_label_map.empty()) {
                std::map<int, FrameEventsTrack> face_obj_id_to_actions_track;
                ConvertActionMapsToFrameEventTracks(face_obj_id_to_action_maps, default_action_index,
                                                    &face_obj_id_to_actions_track);

                const int start_frame = 0;
                const int end_frame = face_obj_id_to_action_maps.size();
                const int smooth_window_size = static_cast<int>(cap->fps() * FLAGS_d_ad);
                const int smooth_min_length = static_cast<int>(cap->fps() * FLAGS_min_ad);
                std::map<int, RangeEventsTrack> face_obj_id_to_events;
                SmoothTracks(face_obj_id_to_actions_track, start_frame, end_frame,
                             smooth_window_size, smooth_min_length, default_action_index,
                             &face_obj_id_to_events);

                slog::debug << "Final ID->events mapping" << slog::endl;
                logger.DumpTracks(face_obj_id_to_events,
                                  actions_map, face_track_id_to_label,
                                  face_id_to_label_map);

                std::vector<std::map<int, int>> face_obj_id_to_smoothed_action_maps;
                ConvertRangeEventsTracksToActionMaps(end_frame, face_obj_id_to_events,
                                                     &face_obj_id_to_smoothed_action_maps);

                slog::debug << "Final per-frame ID->action mapping" << slog::endl;
                logger.DumpDetections(FLAGS_i, frame.size(), work_num_frames,
                                      new_face_tracks,
                                      face_track_id_to_label,
                                      actions_map, face_id_to_label_map,
                                      face_obj_id_to_smoothed_action_maps);
            }
        }

        slog::info << "Metrics report:" << slog::endl;
        metrics.logTotal();
        slog::info << presenter.reportMeans() << slog::endl;
    }
    catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    return 0;
}
