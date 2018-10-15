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

#include "mkldnn/mkldnn_extension_ptr.hpp"

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
void DrawObject(const cv::Rect& rect, const std::string& label,
                const cv::Scalar& text_color, bool plot_bg, cv::Mat* image) {
    cv::rectangle(*image, rect, cv::Scalar(255, 255, 255));

    if (plot_bg) {
        int baseLine = 0;
        const cv::Size label_size =
                cv::getTextSize(label, cv::FONT_HERSHEY_PLAIN, 1, 1, &baseLine);

        cv::rectangle(*image, cv::Point(rect.x, rect.y - label_size.height),
                      cv::Point(rect.x + label_size.width, rect.y + baseLine),
                      cv::Scalar(255, 255, 255), cv::FILLED);
    }

    putText(*image, label, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_PLAIN, 1,
            text_color, 1, cv::LINE_AA);
}

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

    if (FLAGS_m_act.empty()) {
        throw std::logic_error("Parameter -m_act is not set");
    }

    if (FLAGS_m_fd.empty()) {
        throw std::logic_error("Parameter -m_fd is not set");
    }

    if (FLAGS_m_lm.empty()) {
        throw std::logic_error("Parameter -m_lm is not set");
    }

    if (FLAGS_m_reid.empty()) {
        throw std::logic_error("Parameter -m_reid is not set");
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
                    auto extension_ptr = make_so_pointer<MKLDNNPlugin::IMKLDNNExtension>(FLAGS_l);
                    plugin.AddExtension(std::static_pointer_cast<IExtension>(extension_ptr));
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
        action_config.detection_confidence_threshold = FLAGS_t_act;
        ActionDetection action_detector(action_config);

        // Load face detector
        detection::DetectorConfig face_config(fd_model_path, fd_weights_path);
        face_config.plugin = plugins_for_devices[FLAGS_d_fd];
        face_config.is_async = true;
        face_config.confidence_threshold = FLAGS_t_fd;
        face_config.input_h = 600;
        face_config.input_w = 600;
        detection::FaceDetection face_detector(face_config);

        // Load face reid
        CnnConfig reid_config(fr_model_path, fr_weights_path);
        reid_config.max_batch_size = 16;
        reid_config.plugin = plugins_for_devices[FLAGS_d_reid];
        VectorCNN face_reid(reid_config);

        // Load landmarks detector
        CnnConfig landmarks_config(lm_model_path, lm_weights_path);
        landmarks_config.max_batch_size = 16;
        landmarks_config.plugin = plugins_for_devices[FLAGS_d_lm];
        VectorCNN landmarks_detector(landmarks_config);

        // Create face gallery
        EmbeddingsGallery face_gallery(FLAGS_fg, FLAGS_t_reid, landmarks_detector, face_reid);

        // Create tracker for reid
        TrackerParams tracker_reid_params;
        tracker_reid_params.min_track_duration = 1000;
        tracker_reid_params.forget_delay = 150;
        tracker_reid_params.affinity_thr = 0.8;
        tracker_reid_params.min_det_conf = 0.5;
        tracker_reid_params.bbox_heights_range = cv::Vec2f(10, 1080);
        tracker_reid_params.drop_forgotten_tracks = true;
        tracker_reid_params.max_num_objects_in_track = 10000;
        tracker_reid_params.objects_type = "face";

        Tracker tracker_reid(tracker_reid_params);

        // Create Tracker for action recognition
        TrackerParams tracker_action_params;
        tracker_action_params.min_track_duration = 1000;
        tracker_action_params.forget_delay = 150;
        tracker_action_params.affinity_thr = 0.8;
        tracker_action_params.min_det_conf = 0.5;
        tracker_action_params.bbox_heights_range = cv::Vec2f(10, 1080);
        tracker_action_params.drop_forgotten_tracks = true;
        tracker_action_params.max_num_objects_in_track = 10;
        tracker_action_params.objects_type = "action";

        Tracker tracker_action(tracker_action_params);

        cv::Mat frame, prev_frame;
        cv::Mat out_image;
        DetectedActions actions;
        detection::DetectedObjects faces;

        float total_time_ms = 0.f;
        size_t num_frames = 0;
        const char ESC_KEY = 27;
        const cv::Scalar red_color(0, 0, 255);

        float frame_delay = 1000.0f / 30.0f;  // 30 fps is assumed.

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

        while (!is_last_frame) {
            is_last_frame = !cap.GrabNext();
            if (!is_last_frame)
                cap.Retrieve(frame);

            if (!FLAGS_no_show) {
                prev_frame.copyTo(out_image);
            }

            logger.CreateNextFrameRecord(prev_frame_path, prev_fame_idx,
                                         prev_frame.cols, prev_frame.rows);

            auto started = std::chrono::high_resolution_clock::now();

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

            for (size_t i = 0; i < faces.size(); i++) {
                auto id = face_gallery.ComputeId(embeddings[i]);
                tracked_face_objects.emplace_back(faces[i].rect, faces[i].confidence, id);
            }
            tracker_reid.Process(prev_frame, tracked_face_objects, num_frames * frame_delay);

            const auto tracked_faces = tracker_reid.TrackedDetectionsWithLabels();
            for (const auto& face : tracked_faces) {
                if (!FLAGS_no_show) {
                    DrawObject(face.rect, face_gallery.GetLabelById(face.label),
                               red_color, true, &out_image);
                }
                logger.AddFaceToFrame(face.rect, face_gallery.GetLabelById(face.label));
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

                if (!FLAGS_no_show) {
                    DrawObject(action.rect, label_desc, label_color, false, &out_image);
                }
                logger.AddPersonToFrame(action.rect, label_desc);
            }
            logger.FinalizeFrameRecord();

            if (!FLAGS_no_show) {
                cv::putText(out_image,
                            std::to_string(static_cast<int>(1e3f / elapsed_ms)) + " fps",
                            cv::Point(10, 50), CV_FONT_HERSHEY_SIMPLEX, 2,
                            cv::Scalar(0, 0, 255), 2);

                cv::imshow("Smart classroom demo", out_image);
            }

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
