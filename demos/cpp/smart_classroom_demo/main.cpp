// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <chrono>  // NOLINT

#include <gflags/gflags.h>
#include <monitors/presenter.h>
#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>
#include <string>
#include <memory>
#include <limits>
#include <vector>
#include <deque>
#include <map>
#include <set>
#include <algorithm>
#include <utility>
#include <ie_iextension.h>

#include "actions.hpp"
#include "action_detector.hpp"
#include "cnn.hpp"
#include "detector.hpp"
#include "face_reid.hpp"
#include "tracker.hpp"
#include "image_grabber.hpp"
#include "logger.hpp"
#include "smart_classroom_demo.hpp"

using namespace InferenceEngine;

namespace {

class Visualizer {
private:
    cv::Mat frame_;
    cv::Mat top_persons_;
    const bool enabled_;
    const int num_top_persons_;
    cv::VideoWriter& writer_;
    float rect_scale_x_;
    float rect_scale_y_;
    static int const max_input_width_ = 1920;
    std::string const main_window_name_ = "Smart classroom demo";
    std::string const top_window_name_ = "Top-k students";
    static int const crop_width_ = 128;
    static int const crop_height_ = 320;
    static int const header_size_ = 80;
    static int const margin_size_ = 5;

public:
    Visualizer(bool enabled, cv::VideoWriter& writer, int num_top_persons) : enabled_(enabled), num_top_persons_(num_top_persons), writer_(writer),
                                                        rect_scale_x_(0), rect_scale_y_(0) {
        if (!enabled_) {
            return;
        }

        cv::namedWindow(main_window_name_);

        if (num_top_persons_ > 0) {
            cv::namedWindow(top_window_name_);

            CreateTopWindow();
            ClearTopWindow();
        }
    }

    static cv::Size GetOutputSize(const cv::Size& input_size) {
        if (input_size.width > max_input_width_) {
            float ratio = static_cast<float>(input_size.height) / input_size.width;
            return cv::Size(max_input_width_, cvRound(ratio*max_input_width_));
        }
        return input_size;
    }

    void SetFrame(const cv::Mat& frame) {
        if (!enabled_ && !writer_.isOpened()) {
            return;
        }

        frame_ = frame.clone();
        rect_scale_x_ = 1;
        rect_scale_y_ = 1;
        cv::Size new_size = GetOutputSize(frame_.size());
        if (new_size != frame_.size()) {
            rect_scale_x_ = static_cast<float>(new_size.height) / frame_.size().height;
            rect_scale_y_ = static_cast<float>(new_size.width) / frame_.size().width;
            cv::resize(frame_, frame_, new_size);
        }
    }

    void Show() const {
        if (enabled_) {
            cv::imshow(main_window_name_, frame_);
        }

        if (writer_.isOpened()) {
            writer_ << frame_;
        }
    }

    void DrawCrop(cv::Rect roi, int id, const cv::Scalar& color) const {
        if (!enabled_ || num_top_persons_ <= 0) {
            return;
        }

        if (id < 0 || id >= num_top_persons_) {
            return;
        }

        if (rect_scale_x_ != 1 || rect_scale_y_ != 1) {
            roi.x = cvRound(roi.x * rect_scale_x_);
            roi.y = cvRound(roi.y * rect_scale_y_);

            roi.height = cvRound(roi.height * rect_scale_y_);
            roi.width = cvRound(roi.width * rect_scale_x_);
        }

        roi.x = std::max(0, roi.x);
        roi.y = std::max(0, roi.y);
        roi.width = std::min(roi.width, frame_.cols - roi.x);
        roi.height = std::min(roi.height, frame_.rows - roi.y);

        const auto crop_label = std::to_string(id + 1);

        auto frame_crop = frame_(roi).clone();
        cv::resize(frame_crop, frame_crop, cv::Size(crop_width_, crop_height_));

        const int shift = (id + 1) * margin_size_ + id * crop_width_;
        frame_crop.copyTo(top_persons_(cv::Rect(shift, header_size_, crop_width_, crop_height_)));

        cv::imshow(top_window_name_, top_persons_);
    }

    void DrawObject(cv::Rect rect, const std::string& label_to_draw,
                    const cv::Scalar& text_color, const cv::Scalar& bbox_color, bool plot_bg) {
        if (!enabled_ && !writer_.isOpened()) {
            return;
        }

        if (rect_scale_x_ != 1 || rect_scale_y_ != 1) {
            rect.x = cvRound(rect.x * rect_scale_x_);
            rect.y = cvRound(rect.y * rect_scale_y_);

            rect.height = cvRound(rect.height * rect_scale_y_);
            rect.width = cvRound(rect.width * rect_scale_x_);
        }
        cv::rectangle(frame_, rect, bbox_color);

        if (plot_bg && !label_to_draw.empty()) {
            int baseLine = 0;
            const cv::Size label_size =
                cv::getTextSize(label_to_draw, cv::FONT_HERSHEY_PLAIN, 1, 1, &baseLine);
            cv::rectangle(frame_, cv::Point(rect.x, rect.y - label_size.height),
                            cv::Point(rect.x + label_size.width, rect.y + baseLine),
                            bbox_color, cv::FILLED);
        }
        if (!label_to_draw.empty()) {
            cv::putText(frame_, label_to_draw, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_PLAIN, 1,
                        text_color, 1, cv::LINE_AA);
        }
    }

    void DrawFPS(const float fps, const cv::Scalar& color) {
        if (enabled_ && !writer_.isOpened()) {
            cv::putText(frame_,
                        std::to_string(static_cast<int>(fps)) + " fps",
                        cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1,
                        color, 2, cv::LINE_AA);
        }
    }

    void CreateTopWindow() {
        if (!enabled_ || num_top_persons_ <= 0) {
            return;
        }

        const int width = margin_size_ * (num_top_persons_ + 1) + crop_width_ * num_top_persons_;
        const int height = header_size_ + crop_height_ + margin_size_;

        top_persons_.create(height, width, CV_8UC3);
    }

    void ClearTopWindow() {
        if (!enabled_ || num_top_persons_ <= 0) {
            return;
        }

        top_persons_.setTo(cv::Scalar(255, 255, 255));

        for (int i = 0; i < num_top_persons_; ++i) {
            const int shift = (i + 1) * margin_size_ + i * crop_width_;

            cv::rectangle(top_persons_, cv::Point(shift, header_size_),
                          cv::Point(shift + crop_width_, header_size_ + crop_height_),
                          cv::Scalar(128, 128, 128), cv::FILLED);

            const auto label_to_draw = "#" + std::to_string(i + 1);
            int baseLine = 0;
            const auto label_size =
                cv::getTextSize(label_to_draw, cv::FONT_HERSHEY_SIMPLEX, 2, 2, &baseLine);
            const int text_shift = (crop_width_ - label_size.width) / 2;
            cv::putText(top_persons_, label_to_draw,
                        cv::Point(shift + text_shift, label_size.height + baseLine / 2),
                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        }

        cv::imshow(top_window_name_, top_persons_);
    }

    void Finalize() const {
        if (enabled_) {
            cv::destroyWindow(main_window_name_);

            if (num_top_persons_ > 0) {
                cv::destroyWindow(top_window_name_);
            }
        }

        if (writer_.isOpened()) {
            writer_.release();
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

std::vector<std::string> ParseActionLabels(const std::string& in_str) {
    std::vector<std::string> labels;
    std::string label;
    std::istringstream stream_to_split(in_str);

    while (std::getline(stream_to_split, label, ',')) {
      labels.push_back(label);
    }

    return labels;
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
        SCR_CHECK(face_track_id_to_label.count(cur_obj_id) == 0) << " Repeating face tracks";
        face_track_id_to_label[cur_obj_id] = cur_label;
    }
    return face_track_id_to_label;
}

bool checkDynamicBatchSupport(const Core& ie, const std::string& device)  {
    try  {
        if (ie.GetConfig(device, CONFIG_KEY(DYN_BATCH_ENABLED)).as<std::string>() != PluginConfigParams::YES)
            return false;
    }
    catch(const std::exception&)  {
        return false;
    }
    return true;
}

class FaceRecognizer {
public:
    virtual ~FaceRecognizer() = default;

    virtual bool LabelExists(const std::string &label) const = 0;
    virtual std::string GetLabelByID(int id) const = 0;
    virtual std::vector<std::string> GetIDToLabelMap() const = 0;

    virtual std::vector<int> Recognize(const cv::Mat& frame, const detection::DetectedObjects& faces) = 0;

    virtual void PrintPerformanceCounts(
        const std::string &landmarks_device, const std::string &reid_device) = 0;
};

class FaceRecognizerNull : public FaceRecognizer {
public:
    bool LabelExists(const std::string &) const override { return false; }

    std::string GetLabelByID(int) const override {
        return EmbeddingsGallery::unknown_label;
    }

    std::vector<std::string> GetIDToLabelMap() const override { return {}; }

    std::vector<int> Recognize(const cv::Mat&, const detection::DetectedObjects& faces) override {
        return std::vector<int>(faces.size(), EmbeddingsGallery::unknown_id);
    }

    void PrintPerformanceCounts(
        const std::string &, const std::string &) override {}
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
            bool greedy_reid_matching
    )
        : landmarks_detector(landmarks_detector_config),
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

    bool LabelExists(const std::string &label) const override {
        return face_gallery.LabelExists(label);
    }

    std::string GetLabelByID(int id) const override {
        return face_gallery.GetLabelByID(id);
    }

    std::vector<std::string> GetIDToLabelMap() const override {
        return face_gallery.GetIDToLabelMap();
    }

    std::vector<int> Recognize(const cv::Mat& frame, const detection::DetectedObjects& faces) override {
        std::vector<cv::Mat> face_rois;

        for (const auto& face : faces) {
            face_rois.push_back(frame(face.rect));
        }

        std::vector<cv::Mat> landmarks, embeddings;

        landmarks_detector.Compute(face_rois, &landmarks, cv::Size(2, 5));
        AlignFaces(&face_rois, &landmarks);
        face_reid.Compute(face_rois, &embeddings);
        return face_gallery.GetIDsByEmbeddings(embeddings);
    }

    void PrintPerformanceCounts(
            const std::string &landmarks_device, const std::string &reid_device) {
        landmarks_detector.PrintPerformanceCounts(landmarks_device);
        face_reid.PrintPerformanceCounts(reid_device);
    }

private:
    VectorCNN landmarks_detector;
    VectorCNN face_reid;
    EmbeddingsGallery face_gallery;
};

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }

    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }
    if (FLAGS_m_act.empty() && FLAGS_m_fd.empty()) {
        throw std::logic_error("At least one parameter -m_act or -m_fd must be set");
    }

    return true;
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        /** This demo covers 4 certain topologies and cannot be generalized **/
        slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << slog::endl;

        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        const auto video_path = FLAGS_i;
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

        const auto actions_type = FLAGS_teacher_id.empty()
                                      ? FLAGS_a_top > 0 ? TOP_K : STUDENT
                                      : TEACHER;
        const auto actions_map = actions_type == STUDENT
                                     ? ParseActionLabels(FLAGS_student_ac)
                                     : actions_type == TOP_K
                                         ? ParseActionLabels(FLAGS_top_ac)
                                         : ParseActionLabels(FLAGS_teacher_ac);
        const auto num_top_persons = actions_type == TOP_K ? FLAGS_a_top : -1;
        const auto top_action_id = actions_type == TOP_K
                                   ? std::distance(actions_map.begin(), find(actions_map.begin(), actions_map.end(), FLAGS_top_id))
                                   : -1;
        if (actions_type == TOP_K && (top_action_id < 0 || top_action_id >= static_cast<int>(actions_map.size()))) {
            slog::err << "Cannot find target action: " << FLAGS_top_id << slog::endl;
            return 1;
        }

        slog::info << "Reading video '" << video_path << "'" << slog::endl;
        ImageGrabber cap(video_path);
        if (!cap.IsOpened()) {
            slog::err << "Cannot open the video" << slog::endl;
            return 1;
        }

        slog::info << "Loading Inference Engine" << slog::endl;
        Core ie;

        std::vector<std::string> devices = {FLAGS_d_act, FLAGS_d_fd, FLAGS_d_lm,
                                            FLAGS_d_reid};
        std::set<std::string> loadedDevices;

        slog::info << "Device info: " << slog::endl;

        for (const auto &device : devices) {
            if (loadedDevices.find(device) != loadedDevices.end())
                continue;

            std::cout << ie.GetVersions(device) << std::endl;

            /** Load extensions for the CPU device **/
            if ((device.find("CPU") != std::string::npos)) {
                if (!FLAGS_l.empty()) {
                    // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
                    auto extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
                    ie.AddExtension(extension_ptr, "CPU");
                    slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
                }
            } else if (!FLAGS_c.empty()) {
                // Load Extensions for other plugins not CPU
                ie.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}}, "GPU");
            }

            if (device.find("CPU") != std::string::npos) {
                ie.SetConfig({{PluginConfigParams::KEY_DYN_BATCH_ENABLED, PluginConfigParams::YES}}, "CPU");
            } else if (device.find("GPU") != std::string::npos) {
                ie.SetConfig({{PluginConfigParams::KEY_DYN_BATCH_ENABLED, PluginConfigParams::YES}}, "GPU");
            }

            if (FLAGS_pc)
                ie.SetConfig({{PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES}});

            loadedDevices.insert(device);
        }

        std::unique_ptr<AsyncDetection<DetectedAction>> action_detector;
        if (!ad_model_path.empty()) {
            // Load action detector
            ActionDetectorConfig action_config(ad_model_path);
            action_config.deviceName = FLAGS_d_act;
            action_config.ie = ie;
            action_config.is_async = true;
            action_config.detection_confidence_threshold = static_cast<float>(FLAGS_t_ad);
            action_config.action_confidence_threshold = static_cast<float>(FLAGS_t_ar);
            action_config.num_action_classes = actions_map.size();
            action_detector.reset(new ActionDetection(action_config));
        } else {
            action_detector.reset(new NullDetection<DetectedAction>);
        }

        std::unique_ptr<AsyncDetection<detection::DetectedObject>> face_detector;
        if (!fd_model_path.empty()) {
            // Load face detector
            detection::DetectorConfig face_config(fd_model_path);
            face_config.deviceName = FLAGS_d_fd;
            face_config.ie = ie;
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
            face_registration_det_config.deviceName = FLAGS_d_fd;
            face_registration_det_config.ie = ie;
            face_registration_det_config.is_async = false;
            face_registration_det_config.confidence_threshold = static_cast<float>(FLAGS_t_reg_fd);
            face_registration_det_config.increase_scale_x = static_cast<float>(FLAGS_exp_r_fd);
            face_registration_det_config.increase_scale_y = static_cast<float>(FLAGS_exp_r_fd);

            CnnConfig reid_config(fr_model_path);
            reid_config.deviceName = FLAGS_d_reid;
            if (checkDynamicBatchSupport(ie, FLAGS_d_reid))
                reid_config.max_batch_size = 16;
            else
                reid_config.max_batch_size = 1;
            reid_config.ie = ie;

            CnnConfig landmarks_config(lm_model_path);
            landmarks_config.deviceName = FLAGS_d_lm;
            if (checkDynamicBatchSupport(ie, FLAGS_d_lm))
                landmarks_config.max_batch_size = 16;
            else
                landmarks_config.max_batch_size = 1;
            landmarks_config.ie = ie;

            face_recognizer.reset(new FaceRecognizerDefault(
                landmarks_config, reid_config,
                face_registration_det_config,
                FLAGS_fg, FLAGS_t_reid, FLAGS_min_size_fr, FLAGS_crop_gallery, FLAGS_greedy_reid_matching));

            if (actions_type == TEACHER && !face_recognizer->LabelExists(teacher_id)) {
                slog::err << "Teacher id does not exist in the gallery!" << slog::endl;
                return 1;
            }
        } else {
            slog::warn << "Face recognition models are disabled!" << slog::endl;
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
        tracker_action_params.averaging_window_size_for_labels = FLAGS_ss_t > 0
                                                                 ? FLAGS_ss_t
                                                                 : actions_type == TOP_K ? 5 : 1;
        tracker_action_params.bbox_heights_range = cv::Vec2f(10, 2160);
        tracker_action_params.drop_forgotten_tracks = false;
        tracker_action_params.max_num_objects_in_track = std::numeric_limits<int>::max();
        tracker_action_params.objects_type = "action";

        Tracker tracker_action(tracker_action_params);

        cv::Mat frame, prev_frame;

        float work_time_ms = 0.f;
        float wait_time_ms = 0.f;
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

        if (cap.GrabNext()) {
            cap.Retrieve(frame);
        } else {
            slog::err << "Can't read the first frame" << slog::endl;
            return 1;
        }

        if (actions_type != TOP_K) {
            action_detector->enqueue(frame);
            action_detector->submitRequest();
            face_detector->enqueue(frame);
            face_detector->submitRequest();
        }

        prev_frame = frame.clone();

        bool is_last_frame = false;
        bool is_monitoring_enabled = false;
        auto prev_frame_path = cap.GetVideoPath();

        cv::VideoWriter vid_writer;
        if (!FLAGS_out_v.empty()) {
            vid_writer = cv::VideoWriter(FLAGS_out_v, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                         cap.GetFPS(), Visualizer::GetOutputSize(frame.size()));
        }
        Visualizer sc_visualizer(!FLAGS_no_show, vid_writer, num_top_persons);
        DetectionsLogger logger(std::cout, FLAGS_r, FLAGS_ad, FLAGS_al);

        const int smooth_window_size = static_cast<int>(cap.GetFPS() * FLAGS_d_ad);
        const int smooth_min_length = static_cast<int>(cap.GetFPS() * FLAGS_min_ad);

        std::cout << "To close the application, press 'CTRL+C' here";
        if (!FLAGS_no_show) {
            std::cout << " or switch to the output window and press ESC key";
        }
        std::cout << std::endl;

        cv::Size graphSize{static_cast<int>(frame.cols / 4), 60};
        Presenter presenter(FLAGS_u, frame.rows - graphSize.height - 10, graphSize);

        while (!is_last_frame) {
            logger.CreateNextFrameRecord(cap.GetVideoPath(), work_num_frames, prev_frame.cols, prev_frame.rows);
            auto started = std::chrono::high_resolution_clock::now();

            is_last_frame = !cap.GrabNext();
            if (!is_last_frame)
                cap.Retrieve(frame);

            char key = cv::waitKey(1);
            if (key == ESC_KEY) {
                break;
            }
            presenter.handleKey(key);

            presenter.drawGraphs(prev_frame);

            sc_visualizer.SetFrame(prev_frame);

            if (actions_type == TOP_K) {
                if ( (is_monitoring_enabled && key == SPACE_KEY) ||
                     (!is_monitoring_enabled && key != SPACE_KEY) ) {
                    if (key == SPACE_KEY) {
                        action_detector->wait();
                        action_detector->fetchResults();

                        tracker_action.Reset();
                        top_k_obj_ids.clear();

                        is_monitoring_enabled = false;

                        sc_visualizer.ClearTopWindow();
                    }

                    auto elapsed = std::chrono::high_resolution_clock::now() - started;
                    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

                    wait_time_ms += elapsed_ms;
                    ++wait_num_frames;

                    sc_visualizer.DrawFPS(1e3f / (wait_time_ms / static_cast<float>(wait_num_frames) + 1e-6f),
                                          green_color);
                } else {
                    if (key == SPACE_KEY) {
                        is_monitoring_enabled = true;

                        action_detector->enqueue(prev_frame);
                        action_detector->submitRequest();
                    }

                    action_detector->wait();
                    DetectedActions actions = action_detector->fetchResults();

                    if (!is_last_frame) {
                        prev_frame_path = cap.GetVideoPath();
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

                    auto elapsed = std::chrono::high_resolution_clock::now() - started;
                    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

                    work_time_ms += elapsed_ms;
                    ++work_num_frames;

                    sc_visualizer.DrawFPS(1e3f / (work_time_ms / static_cast<float>(work_num_frames) + 1e-6f),
                                          red_color);

                    for (const auto& action : tracked_actions) {
                        auto box_color = white_color;
                        std::string box_caption = "";

                        if (top_k_obj_ids.count(action.object_id) > 0) {
                            box_color = red_color;
                            box_caption = std::to_string(top_k_obj_ids[action.object_id] + 1);
                        }

                        sc_visualizer.DrawObject(action.rect, box_caption, white_color, box_color, true);
                    }
                }
            } else {
                face_detector->wait();
                detection::DetectedObjects faces = face_detector->fetchResults();

                action_detector->wait();
                DetectedActions actions = action_detector->fetchResults();

                if (!is_last_frame) {
                    prev_frame_path = cap.GetVideoPath();
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

                auto elapsed = std::chrono::high_resolution_clock::now() - started;
                auto elapsed_ms =
                        std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

                work_time_ms += elapsed_ms;

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

                sc_visualizer.DrawFPS(1e3f / (work_time_ms / static_cast<float>(work_num_frames) + 1e-6f),
                                      red_color);

                ++work_num_frames;
            }

            ++total_num_frames;

            sc_visualizer.Show();

            if (FLAGS_last_frame >= 0 && work_num_frames > static_cast<size_t>(FLAGS_last_frame)) {
                break;
            }
            prev_frame = frame.clone();
            logger.FinalizeFrameRecord();
        }
        sc_visualizer.Finalize();

        slog::info << slog::endl;
        if (work_num_frames > 0) {
            const float mean_time_ms = work_time_ms / static_cast<float>(work_num_frames);
            slog::info << "Mean FPS: " << 1e3f / mean_time_ms << slog::endl;
        }
        slog::info << "Frames processed: " << total_num_frames << slog::endl;
        if (FLAGS_pc) {
            std::map<std::string, std::string>  mapDevices = getMapFullDevicesNames(ie, devices);
            face_detector->wait();
            action_detector->wait();
            action_detector->printPerformanceCounts(getFullDeviceName(mapDevices, FLAGS_d_act));
            face_detector->printPerformanceCounts(getFullDeviceName(mapDevices, FLAGS_d_fd));
            face_recognizer->PrintPerformanceCounts(
                getFullDeviceName(mapDevices, FLAGS_d_lm),
                getFullDeviceName(mapDevices, FLAGS_d_reid));
        }

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
                std::map<int, RangeEventsTrack> face_obj_id_to_events;
                SmoothTracks(face_obj_id_to_actions_track, start_frame, end_frame,
                             smooth_window_size, smooth_min_length, default_action_index,
                             &face_obj_id_to_events);

                slog::info << "Final ID->events mapping" << slog::endl;
                logger.DumpTracks(face_obj_id_to_events,
                                  actions_map, face_track_id_to_label,
                                  face_id_to_label_map);

                std::vector<std::map<int, int>> face_obj_id_to_smoothed_action_maps;
                ConvertRangeEventsTracksToActionMaps(end_frame, face_obj_id_to_events,
                                                     &face_obj_id_to_smoothed_action_maps);

                slog::info << "Final per-frame ID->action mapping" << slog::endl;
                logger.DumpDetections(cap.GetVideoPath(), frame.size(), work_num_frames,
                                      new_face_tracks,
                                      face_track_id_to_label,
                                      actions_map, face_id_to_label_map,
                                      face_obj_id_to_smoothed_action_maps);
            }
        }

        std::cout << presenter.reportMeans() << '\n';
    }
    catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;

    return 0;
}
