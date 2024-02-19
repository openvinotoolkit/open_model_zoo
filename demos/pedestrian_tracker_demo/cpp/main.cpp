// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gflags/gflags.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

#include <models/detection_model.h>
#include <models/detection_model_centernet.h>
#include <models/detection_model_ssd.h>
#include <models/detection_model_yolo.h>
#include <models/input_data.h>
#include <models/internal_model_data.h>
#include <models/model_base.h>
#include <models/results.h>
#include <monitors/presenter.h>
#include <pipelines/metadata.h>
#include <utils/common.hpp>
#include <utils/config_factory.h>
#include <utils/images_capture.h>
#include <utils/ocv_common.hpp>
#include <utils/performance_metrics.hpp>
#include <utils/slog.hpp>

#include "cnn.hpp"
#include "core.hpp"
#include "descriptor.hpp"
#include "distance.hpp"
#include "pedestrian_tracker_demo.hpp"
#include "tracker.hpp"
#include "utils.hpp"

using ImageWithFrameIndex = std::pair<cv::Mat, int>;

std::unique_ptr<PedestrianTracker> CreatePedestrianTracker(const std::string& reid_model,
                                                           const ov::Core& core,
                                                           const std::string& deviceName,
                                                           bool should_keep_tracking_info) {
    TrackerParams params;

    if (should_keep_tracking_info) {
        params.drop_forgotten_tracks = false;
        params.max_num_objects_in_track = -1;
    }

    std::unique_ptr<PedestrianTracker> tracker(new PedestrianTracker(params));

    // Load reid-model.
    std::shared_ptr<IImageDescriptor> descriptor_fast =
        std::make_shared<ResizedImageDescriptor>(cv::Size(16, 32), cv::InterpolationFlags::INTER_LINEAR);
    std::shared_ptr<IDescriptorDistance> distance_fast = std::make_shared<MatchTemplateDistance>();

    tracker->set_descriptor_fast(descriptor_fast);
    tracker->set_distance_fast(distance_fast);

    if (!reid_model.empty()) {
        ModelConfigTracker reid_config(reid_model);
        reid_config.max_batch_size = 16;  // defaulting to 16
        std::shared_ptr<IImageDescriptor> descriptor_strong =
            std::make_shared<Descriptor>(reid_config, core, deviceName);

        if (descriptor_strong == nullptr) {
            throw std::runtime_error("[SAMPLES] internal error - invalid descriptor");
        }
        std::shared_ptr<IDescriptorDistance> distance_strong = std::make_shared<CosDistance>(descriptor_strong->size());

        tracker->set_descriptor_strong(descriptor_strong);
        tracker->set_distance_strong(distance_strong);
    } else {
        slog::warn << "Reid model "
                   << "was not specified. "
                   << "Only fast reidentification approach will be used." << slog::endl;
    }

    return tracker;
}

bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m_det.empty()) {
        throw std::logic_error("Parameter -m_det is not set");
    }

    if (FLAGS_m_reid.empty()) {
        throw std::logic_error("Parameter -m_reid is not set");
    }

    if (FLAGS_at.empty()) {
        throw std::logic_error("Parameter -at is not set");
    }

    return true;
}

int main(int argc, char** argv) {
    try {
        PerformanceMetrics metrics;

        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        // Reading command line parameters.
        auto det_model = FLAGS_m_det;
        auto reid_model = FLAGS_m_reid;

        auto detlog_out = FLAGS_out;

        auto detector_mode = FLAGS_d_det;
        auto reid_mode = FLAGS_d_reid;

        bool should_print_out = FLAGS_r;

        bool should_show = !FLAGS_no_show;
        int delay = FLAGS_delay;
        if (!should_show)
            delay = -1;
        should_show = (delay >= 0);

        bool should_save_det_log = !detlog_out.empty();

        std::vector<std::string> labels;
        if (!FLAGS_labels.empty())
            labels = DetectionModel::loadLabels(FLAGS_labels);

        std::unique_ptr<ModelBase> detectionModel;
        if (FLAGS_at == "centernet") {
            detectionModel.reset(new ModelCenterNet(det_model, static_cast<float>(FLAGS_t), labels, FLAGS_layout_det));
        } else if (FLAGS_at == "ssd") {
            detectionModel.reset(
                new ModelSSD(det_model, static_cast<float>(FLAGS_t), FLAGS_auto_resize, labels, FLAGS_layout_det));
        } else if (FLAGS_at == "yolo") {
            detectionModel.reset(new ModelYolo(det_model,
                                               static_cast<float>(FLAGS_t),
                                               FLAGS_auto_resize,
                                               FLAGS_yolo_af,
                                               static_cast<float>(FLAGS_iou_t),
                                               labels,
                                               {},
                                               {},
                                               FLAGS_layout_det));
        } else {
            slog::err << "No model type or invalid model type (-at) provided: " + FLAGS_at << slog::endl;
            return -1;
        }

        std::vector<std::string> devices{detector_mode, reid_mode};

        slog::info << ov::get_openvino_version() << slog::endl;
        ov::Core core;

        auto model = detectionModel->compileModel(
            ConfigFactory::getUserConfig(FLAGS_d_det, FLAGS_nireq, FLAGS_nstreams, FLAGS_nthreads),
            core);
        auto req = model.create_infer_request();
        bool should_keep_tracking_info = should_save_det_log || should_print_out;
        std::unique_ptr<PedestrianTracker> tracker =
            CreatePedestrianTracker(reid_model, core, reid_mode, should_keep_tracking_info);

        std::unique_ptr<ImagesCapture> cap =
            openImagesCapture(FLAGS_i,
                              FLAGS_loop,
                              FLAGS_nireq == 1 ? read_type::efficient : read_type::safe,
                              FLAGS_first,
                              FLAGS_read_limit);
        double video_fps = cap->fps();
        if (0.0 == video_fps) {
            // the default frame rate for DukeMTMC dataset
            video_fps = 60.0;
        }

        auto startTime = std::chrono::steady_clock::now();
        cv::Mat frame = cap->read();
        cv::Size firstFrameSize = frame.size();

        LazyVideoWriter videoWriter{FLAGS_o, cap->fps(), FLAGS_limit};
        cv::Size graphSize{static_cast<int>(frame.cols / 4), 60};
        Presenter presenter(FLAGS_u, 10, graphSize);

        for (unsigned frameIdx = 0;; ++frameIdx) {
            detectionModel->preprocess(ImageInputData(frame), req);

            req.infer();

            InferenceResult res;

            res.internalModelData = std::make_shared<InternalImageModelData>(frame.cols, frame.rows);

            res.metaData = std::make_shared<ImageMetaData>(frame, std::chrono::steady_clock::now());

            for (const auto& outName : detectionModel->getOutputsNames()) {
                const auto& outTensor = req.get_tensor(outName);

                if (ov::element::i32 == outTensor.get_element_type()) {
                    res.outputsData.emplace(outName, outTensor);
                } else {
                    res.outputsData.emplace(outName, outTensor);
                }
            }

            auto result = (detectionModel->postprocess(res))->asRef<DetectionResult>();

            TrackedObjects detections;

            for (size_t i = 0; i < result.objects.size(); i++) {
                TrackedObject object;
                object.confidence = result.objects[i].confidence;

                const float frame_width_ = static_cast<float>(frame.cols);
                const float frame_height_ = static_cast<float>(frame.rows);
                object.frame_idx = result.frameId;

                const float x0 = std::min(std::max(0.0f, result.objects[i].x / frame_width_), 1.0f) * frame_width_;
                const float y0 = std::min(std::max(0.0f, result.objects[i].y / frame_height_), 1.0f) * frame_height_;
                const float x1 =
                    std::min(std::max(0.0f, (result.objects[i].x + result.objects[i].width) / frame_width_), 1.0f) *
                    frame_width_;
                const float y1 =
                    std::min(std::max(0.0f, (result.objects[i].y + result.objects[i].height) / frame_height_), 1.0f) *
                    frame_height_;

                object.rect = cv::Rect2f(cv::Point(static_cast<int>(round(static_cast<double>(x0))),
                                                   static_cast<int>(round(static_cast<double>(y0)))),
                                         cv::Point(static_cast<int>(round(static_cast<double>(x1))),
                                                   static_cast<int>(round(static_cast<double>(y1)))));

                if (object.rect.area() > 0 &&
                    (static_cast<int>(result.objects[i].labelID) == FLAGS_person_label || FLAGS_person_label == -1)) {
                    detections.emplace_back(object);
                }
            }

            // timestamp in milliseconds
            uint64_t cur_timestamp = static_cast<uint64_t>(1000.0 / video_fps * frameIdx);
            tracker->Process(frame, detections, cur_timestamp);

            // Drawing colored "worms" (tracks).
            frame = tracker->DrawActiveTracks(frame);

            // Drawing all detected objects on a frame by BLUE COLOR
            for (const auto& detection : detections) {
                cv::rectangle(frame, detection.rect, cv::Scalar(255, 0, 0), 3);
            }

            // Drawing tracked detections only by RED color and print ID and detection
            // confidence level.
            for (const auto& detection : tracker->TrackedDetections()) {
                cv::rectangle(frame, detection.rect, cv::Scalar(0, 0, 255), 3);
                std::string text =
                    std::to_string(detection.object_id) + " conf: " + std::to_string(detection.confidence);
                putHighlightedText(frame,
                                   text,
                                   detection.rect.tl() - cv::Point{10, 10},
                                   cv::FONT_HERSHEY_COMPLEX,
                                   0.65,
                                   cv::Scalar(0, 0, 255),
                                   2);
            }
            presenter.drawGraphs(frame);
            metrics.update(startTime, frame, {10, 22}, cv::FONT_HERSHEY_COMPLEX, 0.65);

            videoWriter.write(frame);
            if (should_show) {
                cv::imshow("dbg", frame);
                char k = cv::waitKey(delay);
                if (k == 27)
                    break;
                presenter.handleKey(k);
            }

            if (should_save_det_log && (frameIdx % 100 == 0)) {
                DetectionLog log = tracker->GetDetectionLog(true);
                SaveDetectionLogToTrajFile(detlog_out, log);
            }
            startTime = std::chrono::steady_clock::now();
            frame = cap->read();
            if (!frame.data)
                break;
            if (frame.size() != firstFrameSize)
                throw std::runtime_error("Can't track objects on images of different size");
        }

        if (should_keep_tracking_info) {
            DetectionLog log = tracker->GetDetectionLog(true);

            if (should_save_det_log)
                SaveDetectionLogToTrajFile(detlog_out, log);
            if (should_print_out)
                PrintDetectionLog(log);
        }

        slog::info << "Metrics report:" << slog::endl;
        metrics.logTotal();
        slog::info << presenter.reportMeans() << slog::endl;
    } catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    } catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    return 0;
}
