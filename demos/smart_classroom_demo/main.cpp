// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <chrono>  // NOLINT

#include <gflags/gflags.h>
#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>
#include <ext_list.hpp>
#include <string>
#include <memory>
#include <limits>
#include <vector>
#include <deque>
#include <map>
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
    const bool enabled_;
    cv::VideoWriter& writer_;
    float rect_scale_x_;
    float rect_scale_y_;
    static int const max_input_width_ = 1920;
    std::string const window_name_ = "Smart classroom demo";

public:
    Visualizer(bool enabled, cv::VideoWriter& writer) : enabled_(enabled), writer_(writer),
                                                        rect_scale_x_(0), rect_scale_y_(0) {
        if (enabled_) {
            cv::namedWindow(window_name_);
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
        if (enabled_ || writer_.isOpened()) {
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
    }

    void Show() const {
        if (enabled_) {
            cv::imshow(window_name_, frame_);
        }
        if (writer_.isOpened()) {
            writer_ << frame_;
        }
    }

    void DrawObject(cv::Rect rect, const std::string& label_to_draw,
                    const cv::Scalar& text_color, const cv::Scalar& bbox_color, bool plot_bg) {
        if (enabled_ || writer_.isOpened()) {
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
                putText(frame_, label_to_draw, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_PLAIN, 1,
                        text_color, 1, cv::LINE_AA);
            }
        }
    }

    void DrawFPS(float fps) {
        if (enabled_ && !writer_.isOpened()) {
            cv::putText(frame_,
                        std::to_string(static_cast<int>(fps)) + " fps",
                        cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 2,
                        cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        }
    }

    void Finalize() const {
        if (enabled_)
            cv::destroyWindow(window_name_);
        if (writer_.isOpened())
            writer_.release();
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
    static std::vector<cv::Scalar> actions_map = {
        cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0), cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 255)};
    if (label < actions_map.size()) {
        return actions_map[label];
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

}  // namespace

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
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

        auto video_path = FLAGS_i;
        auto ad_model_path = FLAGS_m_act;
        auto ad_weights_path = fileNameNoExt(FLAGS_m_act) + ".bin";
        auto fd_model_path = FLAGS_m_fd;
        auto fd_weights_path = fileNameNoExt(FLAGS_m_fd) + ".bin";
        auto fr_model_path = FLAGS_m_reid;
        auto fr_weights_path = fileNameNoExt(FLAGS_m_reid) + ".bin";
        auto lm_model_path = FLAGS_m_lm;
        auto lm_weights_path = fileNameNoExt(FLAGS_m_lm) + ".bin";
        auto teacher_id = FLAGS_teacher_id;

        const auto actions_type = FLAGS_teacher_id == "" ? STUDENT : TEACHER;
        const auto actions_map = actions_type == STUDENT
                                     ? ParseActionLabels(FLAGS_student_ac)
                                     : ParseActionLabels(FLAGS_teacher_ac);

        slog::info << "Reading video '" << video_path << "'" << slog::endl;
        ImageGrabber cap(video_path);
        if (!cap.IsOpened()) {
            slog::err << "Cannot open the video" << slog::endl;
            return 1;
        }

        std::map<std::string, InferencePlugin> plugins_for_devices;
        std::vector<std::string> devices = {FLAGS_d_act, FLAGS_d_fd, FLAGS_d_lm,
                                            FLAGS_d_reid};

        for (const auto &device : devices) {
            if (plugins_for_devices.find(device) != plugins_for_devices.end()) {
                continue;
            }
            slog::info << "Loading plugin " << device << slog::endl;
            InferencePlugin plugin = PluginDispatcher().getPluginByDevice(device);
            printPluginVersion(plugin, std::cout);
            /** Load extensions for the CPU plugin **/
            if ((device.find("CPU") != std::string::npos)) {
                plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());

                if (!FLAGS_l.empty()) {
                    // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
                    auto extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
                    plugin.AddExtension(extension_ptr);
                    slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
                }
            } else if (!FLAGS_c.empty()) {
                // Load Extensions for other plugins not CPU
                plugin.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}});
            }
            if (device.find("CPU") != std::string::npos || device.find("GPU") != std::string::npos) {
                plugin.SetConfig({{PluginConfigParams::KEY_DYN_BATCH_ENABLED, PluginConfigParams::YES}});
            }
            if (FLAGS_pc)
                plugin.SetConfig({{PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES}});
            plugins_for_devices[device] = plugin;
        }

        // Load action detector
        ActionDetectorConfig action_config(ad_model_path, ad_weights_path);
        action_config.plugin = plugins_for_devices[FLAGS_d_act];
        action_config.is_async = true;
        action_config.enabled = !ad_model_path.empty();
        action_config.detection_confidence_threshold = static_cast<float>(FLAGS_t_ad);
        action_config.action_confidence_threshold = static_cast<float>(FLAGS_t_ar);
        action_config.num_action_classes = actions_map.size();
        ActionDetection action_detector(action_config);

        // Load face detector
        detection::DetectorConfig face_config(fd_model_path, fd_weights_path);
        face_config.plugin = plugins_for_devices[FLAGS_d_fd];
        face_config.is_async = true;
        face_config.enabled = !fd_model_path.empty();
        face_config.confidence_threshold = static_cast<float>(FLAGS_t_fd);
        face_config.input_h = FLAGS_inh_fd;
        face_config.input_w = FLAGS_inw_fd;
        face_config.increase_scale_x = static_cast<float>(FLAGS_exp_r_fd);
        face_config.increase_scale_y = static_cast<float>(FLAGS_exp_r_fd);
        detection::FaceDetection face_detector(face_config);

        // Load face reid
        CnnConfig reid_config(fr_model_path, fr_weights_path);
        reid_config.max_batch_size = 16;
        reid_config.enabled = face_config.enabled && !fr_model_path.empty() && !lm_model_path.empty();
        reid_config.plugin = plugins_for_devices[FLAGS_d_reid];
        VectorCNN face_reid(reid_config);

        // Load landmarks detector
        CnnConfig landmarks_config(lm_model_path, lm_weights_path);
        landmarks_config.max_batch_size = 16;
        landmarks_config.enabled = face_config.enabled && reid_config.enabled && !lm_model_path.empty();
        landmarks_config.plugin = plugins_for_devices[FLAGS_d_lm];
        VectorCNN landmarks_detector(landmarks_config);

        // Create face gallery
        EmbeddingsGallery face_gallery(FLAGS_fg, FLAGS_t_reid, landmarks_detector, face_reid);

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
        tracker_action_params.averaging_window_size_for_labels = 1;
        tracker_action_params.bbox_heights_range = cv::Vec2f(10, 2160);
        tracker_action_params.drop_forgotten_tracks = false;
        tracker_action_params.max_num_objects_in_track = std::numeric_limits<int>::max();
        tracker_action_params.objects_type = "action";

        Tracker tracker_action(tracker_action_params);

        cv::Mat frame, prev_frame;
        DetectedActions actions;
        detection::DetectedObjects faces;

        float total_time_ms = 0.f;
        size_t num_frames = 0;
        const char ESC_KEY = 27;
        const cv::Scalar red_color(0, 0, 255);
        const cv::Scalar white_color(255, 255, 255);
        std::vector<std::map<int, int>> face_obj_id_to_action_maps;

        int teacher_track_id = -1;

        if (cap.GrabNext()) {
            cap.Retrieve(frame);
        } else {
            slog::err << "Can't read the first frame" << slog::endl;
            return 1;
        }

        action_detector.enqueue(frame);
        action_detector.submitRequest();
        face_detector.enqueue(frame);
        face_detector.submitRequest();
        prev_frame = frame.clone();

        bool is_last_frame = false;
        auto prev_frame_path = cap.GetVideoPath();

        cv::VideoWriter vid_writer;
        if (!FLAGS_out_v.empty()) {
            vid_writer = cv::VideoWriter(FLAGS_out_v, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                         cap.GetFPS(), Visualizer::GetOutputSize(frame.size()));
        }
        Visualizer sc_visualizer(!FLAGS_no_show, vid_writer);
        DetectionsLogger logger(std::cout, FLAGS_r, FLAGS_ad);

        const int smooth_window_size = static_cast<int>(cap.GetFPS() * FLAGS_d_ad);
        const int smooth_min_length = static_cast<int>(cap.GetFPS() * FLAGS_min_ad);

        if (!FLAGS_no_show) {
            std::cout << "To close the application, press 'CTRL+C' or any key with focus on the output window" << std::endl;
        }
        while (!is_last_frame) {
            logger.CreateNextFrameRecord(cap.GetVideoPath(), num_frames, prev_frame.cols, prev_frame.rows);
            auto started = std::chrono::high_resolution_clock::now();

            is_last_frame = !cap.GrabNext();
            if (!is_last_frame)
                cap.Retrieve(frame);

            sc_visualizer.SetFrame(prev_frame);
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

            std::vector<cv::Mat> face_rois, landmarks, embeddings;
            TrackedObjects tracked_face_objects;

            for (const auto& face : faces) {
                face_rois.push_back(prev_frame(face.rect));
            }
            landmarks_detector.Compute(face_rois, &landmarks, cv::Size(2, 5));
            AlignFaces(&face_rois, &landmarks);
            face_reid.Compute(face_rois, &embeddings);
            auto ids = face_gallery.GetIDsByEmbeddings(embeddings);

            for (size_t i = 0; i < faces.size(); i++) {
                int label = ids.empty() ? EmbeddingsGallery::unknown_id : ids[i];
                tracked_face_objects.emplace_back(faces[i].rect, faces[i].confidence, label);
            }
            tracker_reid.Process(prev_frame, tracked_face_objects, num_frames);

            const auto tracked_faces = tracker_reid.TrackedDetectionsWithLabels();

            TrackedObjects tracked_action_objects;
            for (const auto& action : actions) {
                tracked_action_objects.emplace_back(action.rect, action.detection_conf, action.label);
            }

            tracker_action.Process(prev_frame, tracked_action_objects, num_frames);
            const auto tracked_actions = tracker_action.TrackedDetectionsWithLabels();

            auto elapsed = std::chrono::high_resolution_clock::now() - started;
            auto elapsed_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

            total_time_ms += elapsed_ms;
            num_frames += 1;

            std::map<int, int> frame_face_obj_id_to_action;
            for (size_t j = 0; j < tracked_faces.size(); j++) {
                const auto& face = tracked_faces[j];
                std::string label_to_draw;
                if (face.label != EmbeddingsGallery::unknown_id)
                    label_to_draw += face_gallery.GetLabelByID(face.label);

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
                    logger.AddFaceToFrame(face.rect, face_gallery.GetLabelByID(face.label), "");
                }

                if ((actions_type == TEACHER) && (person_ind >= 0)) {
                    if (face_gallery.GetLabelByID(face.label) == teacher_id) {
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
                    const auto& text_label = face_config.enabled ? "" : action_label;
                    sc_visualizer.DrawObject(action.rect, text_label, action_color, white_color, true);
                    logger.AddPersonToFrame(action.rect, action_label, "");
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

            sc_visualizer.DrawFPS(1e3f / (total_time_ms / static_cast<float>(num_frames) + 1e-6f));
            sc_visualizer.Show();

            char key = cv::waitKey(1);
            if (key == ESC_KEY) {
                break;
            }
            if (FLAGS_last_frame >= 0 && num_frames > static_cast<size_t>(FLAGS_last_frame)) {
                break;
            }
            prev_frame = frame.clone();
            logger.FinalizeFrameRecord();
        }
        sc_visualizer.Finalize();

        slog::info << slog::endl;
        float mean_time_ms = total_time_ms / static_cast<float>(num_frames);
        slog::info << "Mean FPS: " << 1e3f / mean_time_ms << slog::endl;
        slog::info << "Frames processed: " << num_frames << slog::endl;
        if (FLAGS_pc) {
            face_detector.wait();
            action_detector.wait();
            action_detector.PrintPerformanceCounts();
            face_detector.PrintPerformanceCounts();
            face_reid.PrintPerformanceCounts();
            landmarks_detector.PrintPerformanceCounts();
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
                logger.DumpDetections(cap.GetVideoPath(), frame.size(), num_frames,
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

    return 0;
}
