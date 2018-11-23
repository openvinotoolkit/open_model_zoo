/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include <chrono>  // NOLINT

#include <gflags/gflags.h>
#include <opencv2/opencv.hpp>
#include <samples/common.hpp>
#include <ext_list.hpp>
#include <string>
#include <memory>
#include <vector>
#include <map>
#include <ie_iextension.h>

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

public:
    Visualizer(bool enabled, cv::VideoWriter& writer) : enabled_(enabled), writer_(writer) {}

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
            cv::imshow("Smart classroom demo", frame_);
        }
        if (writer_.isOpened()) {
            writer_ << frame_;
        }
    }

    void DrawObject(cv::Rect rect, const std::string& label,
                    const cv::Scalar& text_color, bool plot_bg) {
        if (enabled_ || writer_.isOpened()) {
            if (rect_scale_x_ != 1 || rect_scale_y_ != 1) {
                rect.x = cvRound(rect.x * rect_scale_x_);
                rect.y = cvRound(rect.y * rect_scale_y_);

                rect.height = cvRound(rect.height * rect_scale_y_);
                rect.width = cvRound(rect.width * rect_scale_x_);
            }
            cv::rectangle(frame_, rect, cv::Scalar(255, 255, 255));

            if (plot_bg && label != EmbeddingsGallery::unknown_label) {
                int baseLine = 0;
                const cv::Size label_size =
                    cv::getTextSize(label, cv::FONT_HERSHEY_PLAIN, 1, 1, &baseLine);
                cv::rectangle(frame_, cv::Point(rect.x, rect.y - label_size.height),
                              cv::Point(rect.x + label_size.width, rect.y + baseLine),
                              cv::Scalar(255, 255, 255), cv::FILLED);
            }
            if (label != EmbeddingsGallery::unknown_label) {
                putText(frame_, label, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_PLAIN, 1,
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
};

std::string GetActionTextLabel(const unsigned label) {
    static std::vector<std::string> actions_map = {"sitting", "standing",
                                                   "raising_hand"};
    if (label < actions_map.size()) {
        return actions_map[label];
    }
    return "__undefined__";
}

cv::Scalar GetActionTextColor(const unsigned label) {
    static std::vector<cv::Scalar> actions_map = {
        cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0), cv::Scalar(0, 0, 255)};
    if (label < actions_map.size()) {
        return actions_map[label];
    }
    return cv::Scalar(0, 0, 0);
}
}  // namespace

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }

    std::cout << "[ INFO ] Parsing input parameters" << std::endl;

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
        std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

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

        DetectionsLogger logger(std::cout, FLAGS_r);

        std::cout << "Reading video '" << video_path << "'" << std::endl;
        ImageGrabber cap(video_path);
        if (!cap.IsOpened()) {
            std::cout << "Cannot open the video" << std::endl;
            return 1;
        }

        std::map<std::string, InferencePlugin> plugins_for_devices;
        std::vector<std::string> devices = {FLAGS_d_act, FLAGS_d_fd, FLAGS_d_lm,
                                            FLAGS_d_reid};

        for (const auto &device : devices) {
            if (plugins_for_devices.find(device) != plugins_for_devices.end()) {
                continue;
            }
            std::cout << "Loading plugin " << device << std::endl;
            InferencePlugin plugin = PluginDispatcher({"../../../lib/intel64", ""}).getPluginByDevice(device);
            printPluginVersion(plugin, std::cout);
            /** Load extensions for the CPU plugin **/
            if ((device.find("CPU") != std::string::npos)) {
                plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());

                if (!FLAGS_l.empty()) {
                    // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
                    auto extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
                    plugin.AddExtension(extension_ptr);
                    std::cout << "CPU Extension loaded: " << FLAGS_l << std::endl;
                }
            } else if (!FLAGS_c.empty()) {
                // Load Extensions for other plugins not CPU
                plugin.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}});
            }
            plugin.SetConfig({{PluginConfigParams::KEY_DYN_BATCH_ENABLED, PluginConfigParams::YES}});
            if (FLAGS_pc)
                plugin.SetConfig({{PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES}});
            plugins_for_devices[device] = plugin;
        }

        // Load action detector
        ActionDetectorConfig action_config(ad_model_path, ad_weights_path);
        action_config.plugin = plugins_for_devices[FLAGS_d_act];
        action_config.is_async = true;
        action_config.enabled = !ad_model_path.empty();
        action_config.detection_confidence_threshold = FLAGS_t_act;
        ActionDetection action_detector(action_config);

        // Load face detector
        detection::DetectorConfig face_config(fd_model_path, fd_weights_path);
        face_config.plugin = plugins_for_devices[FLAGS_d_fd];
        face_config.is_async = true;
        face_config.enabled = !fd_model_path.empty();
        face_config.confidence_threshold = FLAGS_t_fd;
        face_config.input_h = FLAGS_inh_fd;
        face_config.input_w = FLAGS_inw_fd;
        face_config.increase_scale_x = FLAGS_exp_r_fd;
        face_config.increase_scale_y = FLAGS_exp_r_fd;
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

        // Create tracker for reid
        TrackerParams tracker_reid_params;
        tracker_reid_params.min_track_duration = 750;
        tracker_reid_params.forget_delay = 150;
        tracker_reid_params.affinity_thr = 0.8;
        tracker_reid_params.bbox_heights_range = cv::Vec2f(10, 1080);
        tracker_reid_params.drop_forgotten_tracks = true;
        tracker_reid_params.max_num_objects_in_track = 10000;
        tracker_reid_params.objects_type = "face";

        Tracker tracker_reid(tracker_reid_params);

        // Create Tracker for action recognition
        TrackerParams tracker_action_params;
        tracker_action_params.min_track_duration = 250;
        tracker_action_params.forget_delay = 150;
        tracker_action_params.affinity_thr = 0.95;
        tracker_action_params.bbox_heights_range = cv::Vec2f(10, 1080);
        tracker_action_params.drop_forgotten_tracks = true;
        tracker_action_params.max_num_objects_in_track = 10;
        tracker_action_params.objects_type = "action";

        Tracker tracker_action(tracker_action_params);

        cv::Mat frame, prev_frame;
        DetectedActions actions;
        detection::DetectedObjects faces;

        float total_time_ms = 0.f;
        size_t num_frames = 0;
        const char ESC_KEY = 27;
        const cv::Scalar red_color(0, 0, 255);
        float frame_delay = 1000.0f / cap.GetFPS();

        if (cap.GrabNext()) {
            cap.Retrieve(frame);
        } else {
            std::cout << "Can't read the first frame" << std::endl;
            return 1;
        }

        action_detector.enqueue(frame);
        action_detector.submitRequest();
        face_detector.enqueue(frame);
        face_detector.submitRequest();
        prev_frame = frame.clone();

        bool is_last_frame = false;
        auto prev_frame_path = cap.GetVideoPath();
        int prev_fame_idx = cap.GetFrameIndex();

        cv::VideoWriter vid_writer(FLAGS_out_v, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                   cap.GetFPS(), Visualizer::GetOutputSize(frame.size()));
        Visualizer sc_visualizer(!FLAGS_no_show, vid_writer);

        while (!is_last_frame) {
            auto started = std::chrono::high_resolution_clock::now();

            is_last_frame = !cap.GrabNext();
            if (!is_last_frame)
                cap.Retrieve(frame);

            sc_visualizer.SetFrame(prev_frame);
            logger.CreateNextFrameRecord(prev_frame_path, prev_fame_idx,
                                         prev_frame.cols, prev_frame.rows);

            face_detector.wait();
            face_detector.fetchResults();
            faces = face_detector.results;

            action_detector.wait();
            action_detector.fetchResults();
            actions = action_detector.results;

            if (!is_last_frame) {
                prev_frame_path = cap.GetVideoPath();
                prev_fame_idx = cap.GetFrameIndex();
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
            tracker_reid.Process(prev_frame, tracked_face_objects, num_frames * frame_delay);

            const auto tracked_faces = tracker_reid.TrackedDetectionsWithLabels();
            for (const auto& face : tracked_faces) {
                sc_visualizer.DrawObject(face.rect, face_gallery.GetLabelByID(face.label),
                                         red_color, true);
                logger.AddFaceToFrame(face.rect, face_gallery.GetLabelByID(face.label));
            }

            TrackedObjects tracked_action_objects;
            for (const auto& action : actions) {
                tracked_action_objects.emplace_back(action.rect, action.detection_conf, action.label);
            }

            tracker_action.Process(prev_frame, tracked_action_objects, num_frames * frame_delay);
            const auto tracked_actions = tracker_action.TrackedDetectionsWithLabels();

            auto elapsed = std::chrono::high_resolution_clock::now() - started;
            auto elapsed_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

            total_time_ms += elapsed_ms;
            num_frames += 1;

            for (const auto& action : tracked_actions) {
                auto label_desc = GetActionTextLabel(action.label);
                auto label_color = GetActionTextColor(action.label);

                sc_visualizer.DrawObject(action.rect, label_desc, label_color, false);
                logger.AddPersonToFrame(action.rect, label_desc);
            }
            logger.FinalizeFrameRecord();
            sc_visualizer.DrawFPS(1e3f / (total_time_ms / static_cast<float>(num_frames) + 1e-6));
            sc_visualizer.Show();

            char key = cv::waitKey(1);
            if (key == ESC_KEY) {
                break;
            }
            prev_frame = frame.clone();
        }

        float mean_time_ms = total_time_ms / static_cast<float>(num_frames);
        std::cout << "Mean FPS: " << 1e3f / mean_time_ms << std::endl;
        if (FLAGS_pc) {
            face_detector.wait();
            action_detector.wait();
            action_detector.PrintPerformanceCounts();
            face_detector.PrintPerformanceCounts();
            face_reid.PrintPerformanceCounts();
            landmarks_detector.PrintPerformanceCounts();
        }
    }
    catch (const std::exception& error) {
        std::cerr << "[ ERROR ] " << error.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
        return 1;
    }

    std::cout << "[ INFO ] Execution successful" << std::endl;

    return 0;
}
