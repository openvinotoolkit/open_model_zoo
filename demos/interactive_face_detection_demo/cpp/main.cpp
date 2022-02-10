// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "interactive_face_detection.hpp"
#include "detectors.hpp"
#include "face.hpp"
#include "visualizer.hpp"
#include <monitors/presenter.h>
#include <utils/images_capture.h>

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validating input arguments--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }
    return true;
}

int main(int argc, char *argv[]) {
    try {
        PerformanceMetrics metrics;

        // ------------------------------ Parsing and validating of input arguments --------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        // --------------------------- 1. Loading Inference Engine -----------------------------
        slog::info << ov::get_openvino_version() << slog::endl;
        ov::Core core;

        FaceDetection faceDetector(FLAGS_m, FLAGS_t, FLAGS_r,
                                   static_cast<float>(FLAGS_bb_enlarge_coef), static_cast<float>(FLAGS_dx_coef), static_cast<float>(FLAGS_dy_coef));
        AgeGenderDetection ageGenderDetector(FLAGS_m_ag, FLAGS_r);
        HeadPoseDetection headPoseDetector(FLAGS_m_hp, FLAGS_r);
        EmotionsDetection emotionsDetector(FLAGS_m_em, FLAGS_r);
        FacialLandmarksDetection facialLandmarksDetector(FLAGS_m_lm, FLAGS_r);
        AntispoofingClassifier antispoofingClassifier(FLAGS_m_am, FLAGS_r);
        // ---------------------------------------------------------------------------------------------------

        // --------------------------- 2. Reading IR models and loading them to plugins ----------------------
        Load(faceDetector).into(core, FLAGS_d);
        Load(ageGenderDetector).into(core, FLAGS_d);
        Load(headPoseDetector).into(core, FLAGS_d);
        Load(emotionsDetector).into(core, FLAGS_d);
        Load(facialLandmarksDetector).into(core, FLAGS_d);
        Load(antispoofingClassifier).into(core, FLAGS_d);
        // ----------------------------------------------------------------------------------------------------

        Timer timer;
        std::ostringstream out;
        size_t framesCounter = 0;
        double msrate = 1000.0 / FLAGS_fps;
        std::list<Face::Ptr> faces;
        size_t id = 0;

        std::unique_ptr<ImagesCapture> cap = openImagesCapture(FLAGS_i, FLAGS_loop);

        auto startTime = std::chrono::steady_clock::now();
        cv::Mat frame = cap->read();
        if (!frame.data) {
            throw std::runtime_error("Can't read an image from the input");
        }

        Presenter presenter(FLAGS_u, 60, {frame.cols / 4, 60});

        Visualizer visualizer{frame.size()};
        if (!FLAGS_no_show_emotion_bar && emotionsDetector.enabled()) {
                visualizer.enableEmotionBar(emotionsDetector.emotionsVec);
        }

        LazyVideoWriter videoWriter{FLAGS_o, cap->fps(), FLAGS_limit};

        // Detecting all faces on the first frame and reading the next one
        faceDetector.submitRequest(frame);

        auto startTimeNextFrame = std::chrono::steady_clock::now();
        cv::Mat nextFrame = cap->read();
        while (frame.data) {
            timer.start("total");
            const auto startTimePrevFrame = startTime;
            cv::Mat prevFrame = std::move(frame);
            startTime = startTimeNextFrame;
            frame = std::move(nextFrame);
            framesCounter++;

            // Retrieving face detection results for the previous frame
            std::vector<FaceDetection::Result> prev_detection_results = faceDetector.fetchResults();

            // No valid frame to infer if previous frame is the last
            if (frame.data) {
                if (frame.size() != prevFrame.size()) {
                    throw std::runtime_error("Images of different size are not supported");
                }
                faceDetector.submitRequest(frame);
            }

            // Filling inputs of face analytics networks
            for (auto &&face : prev_detection_results) {
                cv::Rect clippedRect = face.location & cv::Rect({0, 0}, prevFrame.size());
                const cv::Mat& crop = prevFrame(clippedRect);
                ageGenderDetector.enqueue(crop);
                headPoseDetector.enqueue(crop);
                emotionsDetector.enqueue(crop);
                facialLandmarksDetector.enqueue(crop);
                antispoofingClassifier.enqueue(crop);
            }

            // Running Age/Gender Recognition, Head Pose Estimation, Emotions Recognition, Facial Landmarks Estimation and Antispoofing Classifier networks simultaneously
            ageGenderDetector.submitRequest();
            headPoseDetector.submitRequest();
            emotionsDetector.submitRequest();
            facialLandmarksDetector.submitRequest();
            antispoofingClassifier.submitRequest();

            // Read the next frame while waiting for inference results
            startTimeNextFrame = std::chrono::steady_clock::now();
            nextFrame = cap->read();

            //  Postprocessing
            std::list<Face::Ptr> prev_faces;

            if (!FLAGS_no_smooth) {
                prev_faces.insert(prev_faces.begin(), faces.begin(), faces.end());
            }

            faces.clear();

            // For every detected face
            for (size_t i = 0; i < prev_detection_results.size(); i++) {
                auto& result = prev_detection_results[i];
                cv::Rect rect = result.location & cv::Rect({0, 0}, prevFrame.size());

                Face::Ptr face;
                if (!FLAGS_no_smooth) {
                    face = matchFace(rect, prev_faces);
                    float intensity_mean = calcMean(prevFrame(rect));

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

                face->ageGenderEnable(ageGenderDetector.enabled());
                if (face->isAgeGenderEnabled()) {
                    AgeGenderDetection::Result ageGenderResult = ageGenderDetector[i];
                    face->updateGender(ageGenderResult.maleProb);
                    face->updateAge(ageGenderResult.age);
                }

                face->emotionsEnable(emotionsDetector.enabled());
                if (face->isEmotionsEnabled()) {
                    face->updateEmotions(emotionsDetector[i]);
                }

                face->headPoseEnable(headPoseDetector.enabled());
                if (face->isHeadPoseEnabled()) {
                    face->updateHeadPose(headPoseDetector[i]);
                }

                face->landmarksEnable(facialLandmarksDetector.enabled());
                if (face->isLandmarksEnabled()) {
                    face->updateLandmarks(facialLandmarksDetector[i]);
                }

                face->antispoofingEnable(antispoofingClassifier.enabled());
                if (face->isAntispoofingEnabled()) {
                    face->updateRealFaceConfidence(antispoofingClassifier[i]);
                }

                faces.push_back(face);
            }

            // drawing faces
            visualizer.draw(prevFrame, faces);

            presenter.drawGraphs(prevFrame);
            metrics.update(startTimePrevFrame, prevFrame, { 10, 22 }, cv::FONT_HERSHEY_COMPLEX, 0.65);

            timer.finish("total");

            videoWriter.write(prevFrame);

            int delay = std::max(1, static_cast<int>(msrate - timer["total"].getLastCallDuration()));
            if (!FLAGS_no_show) {
                cv::imshow("Detection results", prevFrame);
                int key = cv::waitKey(delay);
                if (27 == key || 'Q' == key || 'q' == key) {
                    break;
                }
                presenter.handleKey(key);
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
