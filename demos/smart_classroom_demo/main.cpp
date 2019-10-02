// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <chrono>  // NOLINT

#include <gflags/gflags.h>
#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>
#ifdef WITH_EXTENSIONS
#include <ext_list.hpp>
#endif
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
                        cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 2,
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
                        cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
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
    catch(const std::exception& error)  {
        return false;
    }
    return true;
}

}  // namespace

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


int main(int argc, char* argv[]) {
    try {
        /** This demo covers 4 certain topologies and cannot be generalized **/
        slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << slog::endl;

        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        const auto video_path = FLAGS_i;
        const auto ad_model_path = FLAGS_m_act;
        const auto ad_weights_path = fileNameNoExt(FLAGS_m_act) + ".bin";
        const auto fd_model_path = FLAGS_m_fd;
        const auto fd_weights_path = fileNameNoExt(FLAGS_m_fd) + ".bin";
        const auto fr_model_path = FLAGS_m_reid;
        const auto fr_weights_path = fileNameNoExt(FLAGS_m_reid) + ".bin";
        const auto lm_model_path = FLAGS_m_lm;
        const auto lm_weights_path = fileNameNoExt(FLAGS_m_lm) + ".bin";
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
#ifdef WITH_EXTENSIONS
                ie.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>(), "CPU");
#endif

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

        // Load action detector
        ActionDetectorConfig action_config(ad_model_path, ad_weights_path);
        action_config.deviceName = FLAGS_d_act;
        action_config.ie = ie;
        action_config.is_async = true;
        action_config.enabled = !ad_model_path.empty();
        action_config.detection_confidence_threshold = static_cast<float>(FLAGS_t_ad);
        action_config.action_confidence_threshold = static_cast<float>(FLAGS_t_ar);
        action_config.num_action_classes = actions_map.size();
        ActionDetection action_detector(action_config);

        // Load face detector
        detection::DetectorConfig face_config(fd_model_path, fd_weights_path);
        face_config.deviceName = FLAGS_d_fd;
        face_config.ie = ie;
        face_config.is_async = true;
        face_config.enabled = !fd_model_path.empty();
        face_config.confidence_threshold = static_cast<float>(FLAGS_t_fd);
        face_config.input_h = FLAGS_inh_fd;
        face_config.input_w = FLAGS_inw_fd;
        face_config.increase_scale_x = static_cast<float>(FLAGS_exp_r_fd);
        face_config.increase_scale_y = static_cast<float>(FLAGS_exp_r_fd);
        detection::FaceDetection face_detector(face_config);

        // Load face detector for face database registration
        detection::DetectorConfig face_registration_det_config(fd_model_path, fd_weights_path);
        face_registration_det_config.deviceName = FLAGS_d_fd;
        face_registration_det_config.ie = ie;
        face_registration_det_config.enabled = !fd_model_path.empty();
        face_registration_det_config.is_async = false;
        face_registration_det_config.confidence_threshold = static_cast<float>(FLAGS_t_reg_fd);
        face_registration_det_config.increase_scale_x = static_cast<float>(FLAGS_exp_r_fd);
        face_registration_det_config.increase_scale_y = static_cast<float>(FLAGS_exp_r_fd);
        detection::FaceDetection face_detector_for_registration(face_registration_det_config);

        // Load face reid
        CnnConfig reid_config(fr_model_path, fr_weights_path);
        reid_config.enabled = face_config.enabled && !fr_model_path.empty() && !lm_model_path.empty();
        reid_config.deviceName = FLAGS_d_reid;
        if (checkDynamicBatchSupport(ie, FLAGS_d_reid))
            reid_config.max_batch_size = 16;
        else
            reid_config.max_batch_size = 1;
        reid_config.ie = ie;
        VectorCNN face_reid(reid_config);

        // Load landmarks detector
        CnnConfig landmarks_config(lm_model_path, lm_weights_path);
        landmarks_config.max_batch_size = 16;
        landmarks_config.enabled = face_config.enabled && reid_config.enabled && !lm_model_path.empty();
        landmarks_config.deviceName = FLAGS_d_lm;
        if (checkDynamicBatchSupport(ie, FLAGS_d_lm))
            landmarks_config.max_batch_size = 16;
        else
            landmarks_config.max_batch_size = 1;
        landmarks_config.ie = ie;
        VectorCNN landmarks_detector(landmarks_config);

        // Create face gallery
        EmbeddingsGallery face_gallery(FLAGS_fg, FLAGS_t_reid, FLAGS_min_size_fr, FLAGS_crop_gallery,
                                       face_detector_for_registration, landmarks_detector, face_reid);

        if (!reid_config.enabled) {
            slog::warn << "Face recognition models are disabled!"  << slog::endl;
        } else if (!face_gallery.size()) {
            slog::warn << "Face reid gallery is empty!"  << slog::endl;
        } else {
            slog::info << "Face reid gallery size: " << face_gallery.size() << slog::endl;
        }

        if (actions_type == TEACHER && !face_gallery.LabelExists(teacher_id)) {
            slog::err << "Teacher id does not exist in the gallery!"  << slog::endl;
            return 1;
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
        DetectedActions actions;
        detection::DetectedObjects faces;

        float work_time_ms = 0.f;
        size_t work_num_frames = 0;
        size_t total_num_frames = 0;
        const char ESC_KEY = 27;
        const cv::Scalar green_color(0, 255, 0);
        const cv::Scalar red_color(0, 0, 255);
        const cv::Scalar white_color(255, 255, 255);
        std::vector<std::map<int, int>> face_obj_id_to_action_maps;
        std::map<int, int> top_k_obj_ids;

        if (cap.GrabNext()) {
            cap.Retrieve(frame);
        } else {
            slog::err << "Can't read the first frame" << slog::endl;
            return 1;
        }

        if (actions_type != TOP_K) {
            action_detector.enqueue(frame);
            action_detector.submitRequest();
            face_detector.enqueue(frame);
            face_detector.submitRequest();
        }

        prev_frame = frame.clone();

        bool is_last_frame = false;
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

        while (!is_last_frame) {
            logger.CreateNextFrameRecord(cap.GetVideoPath(), work_num_frames, prev_frame.cols, prev_frame.rows);

            is_last_frame = !cap.GrabNext();
            if (!is_last_frame)
                cap.Retrieve(frame);

            char key = cv::waitKey(500);
            if (key == ESC_KEY) {
                break;
            }

            sc_visualizer.SetFrame(prev_frame);

            {
                face_detector.wait();
                face_detector.fetchResults();
                faces = face_detector.results;

                action_detector.wait();
                action_detector.fetchResults();
                actions = action_detector.results;

                if (!is_last_frame) {
                    prev_frame_path = cap.GetVideoPath();
                    face_detector.enqueue(frame);
                    face_detector.submitRequest();
                    action_detector.enqueue(frame);
                    action_detector.submitRequest();
                }

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
            face_detector.wait();
            action_detector.wait();
            action_detector.PrintPerformanceCounts(getFullDeviceName(mapDevices, FLAGS_d_act));
            face_detector.PrintPerformanceCounts(getFullDeviceName(mapDevices, FLAGS_d_fd));
            face_reid.PrintPerformanceCounts(getFullDeviceName(mapDevices, FLAGS_d_reid));
            landmarks_detector.PrintPerformanceCounts(getFullDeviceName(mapDevices, FLAGS_d_lm));
        }

        if (actions_type == STUDENT) {
            auto face_tracks = tracker_reid.vector_tracks();

            // correct labels for track
            std::vector<Track> new_face_tracks = UpdateTrackLabelsToBestAndFilterOutUnknowns(face_tracks);
            std::map<int, int> face_track_id_to_label = GetMapFaceTrackIdToLabel(new_face_tracks);

            if (reid_config.enabled && face_gallery.size() > 0) {
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
                                  face_gallery.GetIDToLabelMap());

                std::vector<std::map<int, int>> face_obj_id_to_smoothed_action_maps;
                ConvertRangeEventsTracksToActionMaps(end_frame, face_obj_id_to_events,
                                                     &face_obj_id_to_smoothed_action_maps);

                slog::info << "Final per-frame ID->action mapping" << slog::endl;
                logger.DumpDetections(cap.GetVideoPath(), frame.size(), work_num_frames,
                                      new_face_tracks,
                                      face_track_id_to_label,
                                      actions_map, face_gallery.GetIDToLabelMap(),
                                      face_obj_id_to_smoothed_action_maps);
            }
        }
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

    return 1;
}
