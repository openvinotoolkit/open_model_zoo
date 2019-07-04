// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core.hpp"
#include "utils.hpp"
#include "tracker.hpp"
#include "descriptor.hpp"
#include "distance.hpp"
#include "detector.hpp"
#include "image_reader.hpp"
#include "pedestrian_tracker_demo.hpp"

#include <opencv2/core.hpp>

#include <iostream>
#include <utility>
#include <vector>
#include <map>
#include <memory>
#include <string>
#include <gflags/gflags.h>

using namespace InferenceEngine;
using ImageWithFrameIndex = std::pair<cv::Mat, int>;

std::unique_ptr<PedestrianTracker>
CreatePedestrianTracker(const std::string& reid_model,
                        const std::string& reid_weights,
                        const InferenceEngine::Core & ie,
                        const std::string & deviceName,
                        bool should_keep_tracking_info) {
    TrackerParams params;

    if (should_keep_tracking_info) {
        params.drop_forgotten_tracks = false;
        params.max_num_objects_in_track = -1;
    }

    std::unique_ptr<PedestrianTracker> tracker(new PedestrianTracker(params));

    // Load reid-model.
    std::shared_ptr<IImageDescriptor> descriptor_fast =
        std::make_shared<ResizedImageDescriptor>(
            cv::Size(16, 32), cv::InterpolationFlags::INTER_LINEAR);
    std::shared_ptr<IDescriptorDistance> distance_fast =
        std::make_shared<MatchTemplateDistance>();

    tracker->set_descriptor_fast(descriptor_fast);
    tracker->set_distance_fast(distance_fast);

    if (!reid_model.empty() && !reid_weights.empty()) {
        CnnConfig reid_config(reid_model, reid_weights);
        reid_config.max_batch_size = 16;

        std::shared_ptr<IImageDescriptor> descriptor_strong =
            std::make_shared<DescriptorIE>(reid_config, ie, deviceName);

        if (descriptor_strong == nullptr) {
            THROW_IE_EXCEPTION << "[SAMPLES] internal error - invalid descriptor";
        }
        std::shared_ptr<IDescriptorDistance> distance_strong =
            std::make_shared<CosDistance>(descriptor_strong->size());

        tracker->set_descriptor_strong(descriptor_strong);
        tracker->set_distance_strong(distance_strong);
    } else {
        std::cout << "WARNING: Either reid model or reid weights "
            << "were not specified. "
            << "Only fast reidentification approach will be used." << std::endl;
    }

    return tracker;
}

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }

    std::cout << "Parsing input parameters" << std::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m_det.empty()) {
        throw std::logic_error("Parameter -m_det is not set");
    }

    if (FLAGS_m_reid.empty()) {
        throw std::logic_error("Parameter -m_reid is not set");
    }

    return true;
}

int main_work(int argc, char **argv) {
    std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

    if (!ParseAndCheckCommandLine(argc, argv)) {
        return 0;
    }


    // Reading command line parameters.
    auto video_path = FLAGS_i;

    auto det_model = FLAGS_m_det;
    auto det_weights = fileNameNoExt(FLAGS_m_det) + ".bin";

    auto reid_model = FLAGS_m_reid;
    auto reid_weights = fileNameNoExt(FLAGS_m_reid) + ".bin";

    auto detlog_out = FLAGS_out;

    auto detector_mode = FLAGS_d_det;
    auto reid_mode = FLAGS_d_reid;

    auto custom_cpu_library = FLAGS_l;
    auto path_to_custom_layers = FLAGS_c;
    bool should_use_perf_counter = FLAGS_pc;

    bool should_print_out = FLAGS_r;

    bool should_show = !FLAGS_no_show;
    int delay = FLAGS_delay;
    if (!should_show)
        delay = -1;
    should_show = (delay >= 0);

    int first_frame = FLAGS_first;
    int last_frame = FLAGS_last;

    bool should_save_det_log = !detlog_out.empty();

    if (first_frame >= 0)
        std::cout << "first_frame = " << first_frame << std::endl;
    if (last_frame >= 0)
        std::cout << "last_frame = " << last_frame << std::endl;

    std::vector<std::string> devices{detector_mode, reid_mode};
    InferenceEngine::Core ie =
        LoadInferenceEngine(
            devices, custom_cpu_library, path_to_custom_layers,
            should_use_perf_counter);

    DetectorConfig detector_confid(det_model, det_weights);
    ObjectDetector pedestrian_detector(detector_confid, ie, detector_mode);

    bool should_keep_tracking_info = should_save_det_log || should_print_out;
    std::unique_ptr<PedestrianTracker> tracker =
        CreatePedestrianTracker(reid_model, reid_weights, ie, reid_mode,
                                should_keep_tracking_info);


    // Opening video.
    std::unique_ptr<ImageReader> video =
        ImageReader::CreateImageReaderForPath(video_path);

    PT_CHECK(video->IsOpened()) << "Failed to open video: " << video_path;
    double video_fps = video->GetFrameRate();

    if (first_frame > 0)
        video->SetFrameIndex(first_frame);

    std::cout << "To close the application, press 'CTRL+C' here";
    if (!FLAGS_no_show) {
        std::cout << " or switch to the output window and press ESC key";
    }
    std::cout << std::endl;

    for (;;) {
        auto pair = video->Read();
        cv::Mat frame = pair.first;
        int frame_idx = pair.second;

        if (frame.empty()) break;

        PT_CHECK(frame_idx >= first_frame);

        if ( (last_frame >= 0) && (frame_idx > last_frame) ) {
            std::cout << "Frame " << frame_idx << " is greater than last_frame = "
                << last_frame << " -- break";
            break;
        }

        pedestrian_detector.submitFrame(frame, frame_idx);
        pedestrian_detector.waitAndFetchResults();

        TrackedObjects detections = pedestrian_detector.getResults();

        // timestamp in milliseconds
        uint64_t cur_timestamp = static_cast<uint64_t >(1000.0 / video_fps * frame_idx);
        tracker->Process(frame, detections, cur_timestamp);

        if (should_show) {
            // Drawing colored "worms" (tracks).
            frame = tracker->DrawActiveTracks(frame);

            // Drawing all detected objects on a frame by BLUE COLOR
            for (const auto &detection : detections) {
                cv::rectangle(frame, detection.rect, cv::Scalar(255, 0, 0), 3);
            }

            // Drawing tracked detections only by RED color and print ID and detection
            // confidence level.
            for (const auto &detection : tracker->TrackedDetections()) {
                cv::rectangle(frame, detection.rect, cv::Scalar(0, 0, 255), 3);
                std::string text = std::to_string(detection.object_id) +
                    " conf: " + std::to_string(detection.confidence);
                cv::putText(frame, text, detection.rect.tl(), cv::FONT_HERSHEY_COMPLEX,
                            1.0, cv::Scalar(0, 0, 255), 3);
            }

            cv::resize(frame, frame, cv::Size(), 0.5, 0.5);
            cv::imshow("dbg", frame);
            char k = cv::waitKey(delay);
            if (k == 27)
                break;
        }

        if (should_save_det_log && (frame_idx % 100 == 0)) {
            DetectionLog log = tracker->GetDetectionLog(true);
            SaveDetectionLogToTrajFile(detlog_out, log);
        }
    }

    if (should_keep_tracking_info) {
        DetectionLog log = tracker->GetDetectionLog(true);

        if (should_save_det_log)
            SaveDetectionLogToTrajFile(detlog_out, log);
        if (should_print_out)
            PrintDetectionLog(log);
    }
    if (should_use_perf_counter) {
        pedestrian_detector.PrintPerformanceCounts(getFullDeviceName(ie, FLAGS_d_det));
        tracker->PrintReidPerformanceCounts(getFullDeviceName(ie, FLAGS_d_reid));
    }
    return 0;
}

int main(int argc, char **argv) {
    try {
        main_work(argc, argv);
    }
    catch (const std::exception& error) {
        std::cerr << "[ ERROR ] " << error.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
        return 1;
    }

    std::cout << "Execution successful" << std::endl;

    return 0;
}
