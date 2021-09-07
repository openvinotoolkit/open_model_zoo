// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief The entry point for the Inference Engine interactive_face_detection demo application
* \file interactive_face_detection_demo/main.cpp
* \example interactive_face_detection_demo/main.cpp
*/
#include <gflags/gflags.h>
#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <iterator>
#include <map>
#include <list>
#include <set>

#include <inference_engine.hpp>

#include <monitors/presenter.h>
#include <utils/images_capture.h>
#include <utils/ocv_common.hpp>
#include <utils/slog.hpp>

#include "interactive_face_detection.hpp"
#include "detectors.hpp"
#include "face.hpp"
#include "visualizer.hpp"

using namespace InferenceEngine;


bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validating input arguments--------------------------------------
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

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    if (FLAGS_n_ag < 1) {
        throw std::logic_error("Parameter -n_ag cannot be 0");
    }

    if (FLAGS_n_hp < 1) {
        throw std::logic_error("Parameter -n_hp cannot be 0");
    }
    return true;
}

int main(int argc, char *argv[]) {
    try {
        std::cout << "InferenceEngine: " << printable(*GetInferenceEngineVersion()) << std::endl;

        // ------------------------------ Parsing and validating of input arguments --------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        // --------------------------- 1. Loading Inference Engine -----------------------------

        Core ie;

        std::set<std::string> loadedDevices;
        std::pair<std::string, std::string> cmdOptions[] = {
            {FLAGS_d, FLAGS_m},
            {FLAGS_d_ag, FLAGS_m_ag},
            {FLAGS_d_hp, FLAGS_m_hp},
            {FLAGS_d_em, FLAGS_m_em},
            {FLAGS_d_lm, FLAGS_m_lm},
            {FLAGS_d_am, FLAGS_m_am},
        };
        FaceDetection faceDetector(FLAGS_m, FLAGS_d, 1, false, FLAGS_async, FLAGS_t, FLAGS_r,
                                   static_cast<float>(FLAGS_bb_enlarge_coef), static_cast<float>(FLAGS_dx_coef), static_cast<float>(FLAGS_dy_coef));
        AgeGenderDetection ageGenderDetector(FLAGS_m_ag, FLAGS_d_ag, FLAGS_n_ag, FLAGS_dyn_ag, FLAGS_async, FLAGS_r);
        HeadPoseDetection headPoseDetector(FLAGS_m_hp, FLAGS_d_hp, FLAGS_n_hp, FLAGS_dyn_hp, FLAGS_async, FLAGS_r);
        EmotionsDetection emotionsDetector(FLAGS_m_em, FLAGS_d_em, FLAGS_n_em, FLAGS_dyn_em, FLAGS_async, FLAGS_r);
        FacialLandmarksDetection facialLandmarksDetector(FLAGS_m_lm, FLAGS_d_lm, FLAGS_n_lm, FLAGS_dyn_lm, FLAGS_async, FLAGS_r);
        AntispoofingClassifier antispoofingClassifier(FLAGS_m_am, FLAGS_d_am, FLAGS_n_am, FLAGS_dyn_am, FLAGS_async, FLAGS_r);

        for (auto && option : cmdOptions) {
            auto deviceName = option.first;
            auto networkName = option.second;

            if (deviceName.empty() || networkName.empty()) {
                continue;
            }

            if (loadedDevices.find(deviceName) != loadedDevices.end()) {
                continue;
            }
            slog::info << "Loading device " << deviceName << slog::endl;
            slog::info << printable(ie.GetVersions(deviceName)) << slog::endl;

            /** Loading extensions for the CPU device **/
            if ((deviceName.find("CPU") != std::string::npos)) {

                if (!FLAGS_l.empty()) {
                    // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
                    auto extension_ptr = std::make_shared<Extension>(FLAGS_l);
                    ie.AddExtension(extension_ptr, "CPU");
                    slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
                }
            } else if (!FLAGS_c.empty()) {
                // Loading extensions for GPU
                ie.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}}, "GPU");
            }

            loadedDevices.insert(deviceName);
        }

        /** Per-layer metrics **/
        if (FLAGS_pc) {
            ie.SetConfig({{PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES}});
        }
        // ---------------------------------------------------------------------------------------------------

        // --------------------------- 2. Reading IR models and loading them to plugins ----------------------
        // Disable dynamic batching for face detector as it processes one image at a time
        Load(faceDetector).into(ie, FLAGS_d, false);
        Load(ageGenderDetector).into(ie, FLAGS_d_ag, FLAGS_dyn_ag);
        Load(headPoseDetector).into(ie, FLAGS_d_hp, FLAGS_dyn_hp);
        Load(emotionsDetector).into(ie, FLAGS_d_em, FLAGS_dyn_em);
        Load(facialLandmarksDetector).into(ie, FLAGS_d_lm, FLAGS_dyn_lm);
        Load(antispoofingClassifier).into(ie, FLAGS_d_am, FLAGS_dyn_am);
        // ----------------------------------------------------------------------------------------------------

        bool isFaceAnalyticsEnabled = ageGenderDetector.enabled() || headPoseDetector.enabled() ||
                                      emotionsDetector.enabled() || facialLandmarksDetector.enabled() || antispoofingClassifier.enabled();

        Timer timer;
        std::ostringstream out;
        size_t framesCounter = 0;
        double msrate = 1000.0 / FLAGS_fps;
        std::list<Face::Ptr> faces;
        size_t id = 0;

        std::unique_ptr<ImagesCapture> cap = openImagesCapture(FLAGS_i, FLAGS_loop);
        cv::Mat frame = cap->read();
        if (!frame.data) {
            throw std::runtime_error("Can't read an image from the input");
        }

        const cv::Point THROUGHPUT_METRIC_POSITION{10, 45};
        Presenter presenter(FLAGS_u, THROUGHPUT_METRIC_POSITION.y + 15, {frame.cols / 4, 60});

        Visualizer visualizer{frame.size()};
        if (!FLAGS_no_show_emotion_bar && emotionsDetector.enabled()) {
                visualizer.enableEmotionBar(emotionsDetector.emotionsVec);
        }

        cv::VideoWriter videoWriter;
        if (!FLAGS_o.empty() && !videoWriter.open(FLAGS_o, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                                  !FLAGS_no_show && FLAGS_fps > 0.0 ? FLAGS_fps : cap->fps(),
                                                  frame.size())) {
            throw std::runtime_error("Can't open video writer");
        }

        // Detecting all faces on the first frame and reading the next one
        faceDetector.enqueue(frame);
        faceDetector.submitRequest();

        cv::Mat next_frame = cap->read();

        std::cout << "To close the application, press 'CTRL+C' here";
        if (!FLAGS_no_show) {
            std::cout << " or switch to the output window and press Q or Esc";
        }
        std::cout << std::endl;

        while (frame.data) {
            timer.start("total");
            cv::Mat prev_frame = std::move(frame);
            frame = std::move(next_frame);
            framesCounter++;

            // Retrieving face detection results for the previous frame
            faceDetector.wait();
            faceDetector.fetchResults();
            auto prev_detection_results = faceDetector.results;

            // No valid frame to infer if previous frame is the last
            if (frame.data) {
                if (frame.size() != prev_frame.size()) {
                    throw std::runtime_error("Images of different size are not supported");
                }
                faceDetector.enqueue(frame);
                faceDetector.submitRequest();
            }

            // Filling inputs of face analytics networks
            for (auto &&face : prev_detection_results) {
                if (isFaceAnalyticsEnabled) {
                    cv::Rect clippedRect = face.location & cv::Rect({0, 0}, prev_frame.size());
                    cv::Mat face = prev_frame(clippedRect);
                    ageGenderDetector.enqueue(face);
                    headPoseDetector.enqueue(face);
                    emotionsDetector.enqueue(face);
                    facialLandmarksDetector.enqueue(face);
                    antispoofingClassifier.enqueue(face);
                }
            }

            // Running Age/Gender Recognition, Head Pose Estimation, Emotions Recognition, Facial Landmarks Estimation and Antispoofing Classifier networks simultaneously
            if (isFaceAnalyticsEnabled) {
                ageGenderDetector.submitRequest();
                headPoseDetector.submitRequest();
                emotionsDetector.submitRequest();
                facialLandmarksDetector.submitRequest();
                antispoofingClassifier.submitRequest();
            }

            // Read the next frame while waiting for inference results
            next_frame = cap->read();

            if (isFaceAnalyticsEnabled) {
                ageGenderDetector.wait();
                headPoseDetector.wait();
                emotionsDetector.wait();
                facialLandmarksDetector.wait();
                antispoofingClassifier.wait();
            }

            //  Postprocessing
            std::list<Face::Ptr> prev_faces;

            if (!FLAGS_no_smooth) {
                prev_faces.insert(prev_faces.begin(), faces.begin(), faces.end());
            }

            faces.clear();

            // For every detected face
            for (size_t i = 0; i < prev_detection_results.size(); i++) {
                auto& result = prev_detection_results[i];
                cv::Rect rect = result.location & cv::Rect({0, 0}, prev_frame.size());

                Face::Ptr face;
                if (!FLAGS_no_smooth) {
                    face = matchFace(rect, prev_faces);
                    float intensity_mean = calcMean(prev_frame(rect));

                    if ((face == nullptr) ||
                        ((std::abs(intensity_mean - face->_intensity_mean) / face->_intensity_mean) > 0.07f)) {
                        face = std::make_shared<Face>(id++, rect);
                    } else {
                        prev_faces.remove(face);
                    }

                    face->_intensity_mean = intensity_mean;
                    face->_location = rect;
                } else {
                    face = std::make_shared<Face>(id++, rect);
                }

                face->ageGenderEnable((ageGenderDetector.enabled() &&
                                       i < ageGenderDetector.maxBatch));
                if (face->isAgeGenderEnabled()) {
                    AgeGenderDetection::Result ageGenderResult = ageGenderDetector[i];
                    face->updateGender(ageGenderResult.maleProb);
                    face->updateAge(ageGenderResult.age);
                }

                face->emotionsEnable((emotionsDetector.enabled() &&
                                      i < emotionsDetector.maxBatch));
                if (face->isEmotionsEnabled()) {
                    face->updateEmotions(emotionsDetector[i]);
                }

                face->headPoseEnable((headPoseDetector.enabled() &&
                                      i < headPoseDetector.maxBatch));
                if (face->isHeadPoseEnabled()) {
                    face->updateHeadPose(headPoseDetector[i]);
                }

                face->landmarksEnable((facialLandmarksDetector.enabled() &&
                                       i < facialLandmarksDetector.maxBatch));
                if (face->isLandmarksEnabled()) {
                    face->updateLandmarks(facialLandmarksDetector[i]);
                }

                face->antispoofingEnable((antispoofingClassifier.enabled() &&
                    i < antispoofingClassifier.maxBatch));
                if (face->isAntispoofingEnabled()) {
                    face->updateRealFaceConfidence(antispoofingClassifier[i]);
                }

                faces.push_back(face);
            }

            presenter.drawGraphs(prev_frame);

            // drawing faces
            visualizer.draw(prev_frame, faces);

            timer.finish("total");
            out.str("");
            out << "Total image throughput: " << std::fixed << std::setprecision(1)
                << 1000.0 / (timer["total"].getSmoothedDuration()) << " fps";
            cv::putText(prev_frame, out.str(), THROUGHPUT_METRIC_POSITION, cv::FONT_HERSHEY_TRIPLEX, 1,
                        cv::Scalar(255, 0, 0), 2);

            if (videoWriter.isOpened() && (FLAGS_limit == 0 || framesCounter <= FLAGS_limit)) {
                videoWriter.write(prev_frame);
            }

            int delay = std::max(1, static_cast<int>(msrate - timer["total"].getLastCallDuration()));
            if (!FLAGS_no_show) {
                cv::imshow("Detection results", prev_frame);
                int key = cv::waitKey(delay);
                if (27 == key || 'Q' == key || 'q' == key) {
                    break;
                }
                presenter.handleKey(key);
            }
        }

        slog::info << "Number of processed frames: " << framesCounter << slog::endl;
        slog::info << "Total image throughput: " << framesCounter * (1000.0 / timer["total"].getTotalDuration()) << " fps" << slog::endl;

        // Showing performance results
        if (FLAGS_pc) {
            faceDetector.printPerformanceCounts(getFullDeviceName(ie, FLAGS_d));
            ageGenderDetector.printPerformanceCounts(getFullDeviceName(ie, FLAGS_d_ag));
            headPoseDetector.printPerformanceCounts(getFullDeviceName(ie, FLAGS_d_hp));
            emotionsDetector.printPerformanceCounts(getFullDeviceName(ie, FLAGS_d_em));
            facialLandmarksDetector.printPerformanceCounts(getFullDeviceName(ie, FLAGS_d_lm));
            antispoofingClassifier.printPerformanceCounts(getFullDeviceName(ie, FLAGS_d_am));
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
