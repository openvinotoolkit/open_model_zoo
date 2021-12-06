// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "core.hpp"
#include "utils.hpp"
#include "tracker.hpp"
#include "descriptor.hpp"
#include "distance.hpp"
#include "pedestrian_tracker_demo.hpp"

#include <monitors/presenter.h>
#include <utils/images_capture.h>
#include <utils/slog.hpp>
#include <opencv2/core.hpp>

#include <iostream>
#include <utility>
#include <vector>
#include <map>
#include <memory>
#include <string>
#include <gflags/gflags.h>

#include <models/detection_model_centernet.h>
#include <models/detection_model_ssd.h>
#include <models/detection_model_yolo.h>
#include <models/landmarks_model.h>
#include <pipelines/metadata.h>

using ImageWithFrameIndex = std::pair<cv::Mat, int>;


bool ParseAndCheckCommandLine(int argc, char *argv[]) {
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

    if (FLAGS_at.empty()) {
        throw std::logic_error("Parameter -at is not set");
    }

    return true;
}

int main(int argc, char **argv) {
    try {
        PerformanceMetrics metrics;

        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        // Reading command line parameters.
        auto det_model = FLAGS_m_det;
        bool should_show = !FLAGS_no_show;
        int delay = FLAGS_delay;
        if (!should_show)
            delay = -1;
        should_show = (delay >= 0);

        std::string postprocessKey = "heatmap";

        std::unique_ptr<ModelBase> detectionModel;
        if (FLAGS_at == "landmarks") {
            detectionModel.reset(new LandmarksModel(det_model, false, postprocessKey));
        }
        else {
            slog::err << "No model type or invalid model type (-at) provided: " + FLAGS_at << slog::endl;
            return -1;
        }

      

        slog::info << *InferenceEngine::GetInferenceEngineVersion() << slog::endl;
        InferenceEngine::Core ie;
        slog::info << "start load model" << slog::endl;
        auto execNet = detectionModel->loadExecutableNetwork(
            ConfigFactory::getUserConfig(FLAGS_d_det, FLAGS_l, FLAGS_c, FLAGS_nireq, FLAGS_nstreams, FLAGS_nthreads), ie);
        slog::info << "LOADED MODEL" << slog::endl;
        auto req = std::make_shared<InferenceEngine::InferRequest>(execNet.CreateInferRequest());
        slog::info << "CREATED INFER REQUEST" << slog::endl;
        std::unique_ptr<ImagesCapture> cap = openImagesCapture(FLAGS_i, FLAGS_loop, FLAGS_first, FLAGS_read_limit);
        double video_fps = cap->fps();
        if (0.0 == video_fps) {
            // the default frame rate for DukeMTMC dataset
            video_fps = 60.0;
        }
        cv::Mat frame = cap->read();
        if (!frame.data) throw std::runtime_error("Can't read an image from the input");
        cv::Size firstFrameSize = frame.size();

        cv::VideoWriter videoWriter;
        if (!FLAGS_o.empty() && !videoWriter.open(FLAGS_o, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                                  cap->fps(), firstFrameSize)) {
            throw std::runtime_error("Can't open video writer");
        }
        uint32_t framesProcessed = 0;
        cv::Size graphSize{static_cast<int>(frame.cols / 4), 60};
        Presenter presenter(FLAGS_u, 10, graphSize);
        for (unsigned frameIdx = 0; ; ++frameIdx) {
            
            detectionModel->preprocess(ImageInputData(frame), req);
            req->Infer();

            InferenceResult res;

            res.internalModelData = std::make_shared<InternalImageModelData>(frame.cols, frame.rows);

            res.metaData = std::make_shared<ImageMetaData>(frame, std::chrono::steady_clock::now());

            for (const auto& outName : detectionModel->getOutputsNames()) {

                auto blobPtr = req->GetBlob(outName);

                if (InferenceEngine::Precision::I32 == blobPtr->getTensorDesc().getPrecision()) {
                    res.outputsData.emplace(outName,
                        std::make_shared<InferenceEngine::TBlob<int>>(*InferenceEngine::as<InferenceEngine::TBlob<int>>(blobPtr)));
                }
                else {
                    res.outputsData.emplace(outName,
                        std::make_shared<InferenceEngine::TBlob<float>>(*InferenceEngine::as<InferenceEngine::TBlob<float>>(blobPtr)));
                }
            }


            auto result = (detectionModel->postprocess(res))->asRef<LandmarksResult>();
            std::vector<cv::Point2f> landmarks = result.coordinates;
            cv::Size s = frame.size();
            int cols = s.width;
            int lmRadius = static_cast<int>(0.003* cols + 1);
            for (size_t i = 0; i < landmarks.size(); ++i) {
                cv::circle(frame, landmarks[i], lmRadius, cv::Scalar(0, 255, 255), -1);
            }
            //for (auto const& point : landmarks)
                cv::circle(frame, landmarks[60], lmRadius, cv::Scalar(24, 80, 209), -1);
            presenter.drawGraphs(frame);
           
            framesProcessed++;
            if (videoWriter.isOpened() && (FLAGS_limit == 0 || framesProcessed <= FLAGS_limit)) {
                videoWriter.write(frame);
            }
            if (should_show) {
                cv::imshow("dbg", frame);
                char k = cv::waitKey(0);
                if (k == 50)
                    break;
                presenter.handleKey(k);
            }

            frame = cap->read();
            if (!frame.data) break;
            if (frame.size() != firstFrameSize)
                throw std::runtime_error("Can't track objects on images of different size");
     }


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
