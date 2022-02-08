// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "interactive_face_detection.hpp"
#include "detectors.hpp"
#include "face.hpp"
#include "visualizer.hpp"
#include <monitors/presenter.h>
#include <utils/images_capture.h>
#include <utils/default_flags.hpp>

#include <gflags/gflags.h>
#include <iostream>
#include <limits>

namespace {
DEFINE_INPUT_FLAGS
DEFINE_OUTPUT_FLAGS

constexpr char help_msg[] = "show this help message and exit";
DEFINE_bool(h, false, help_msg);

constexpr char face_detection_model_msg[] = "Path to an .xml file with a trained Face Detection model";
DEFINE_string(m, "", face_detection_model_msg);

constexpr char age_gender_model_msg[] = "Path to an .xml file with a trained Age/Gender Recognition model";
DEFINE_string(m_ag, "", age_gender_model_msg);

constexpr char head_pose_model_msg[] = "Path to an .xml file with a trained Head Pose Estimation model";
DEFINE_string(m_hp, "", head_pose_model_msg);

constexpr char emotions_model_msg[] = "Path to an .xml file with a trained Emotions Recognition model";
DEFINE_string(m_em, "", emotions_model_msg);

constexpr char facial_landmarks_model_msg[] = "Path to an .xml file with a trained Facial Landmarks Estimation model";
DEFINE_string(m_lm, "", facial_landmarks_model_msg);

constexpr char antispoofing_model_msg[] = "Path to an .xml file with a trained Antispoofing Classification model";
DEFINE_string(m_am, "", antispoofing_model_msg);

constexpr char device_msg[] =
    "Specify a target device to infer on (the list of available devices is shown below). "
    "Use \"-d HETERO:<comma-separated_devices_list>\" format to specify "
    "HETERO plugin. "
    "Use \"-d MULTI:<comma-separated_devices_list>\" format to specify MULTI plugin. "
    "The application looks for a suitable plugin for the specified device."
    "Default is CPU";
DEFINE_string(d, "CPU", device_msg);

constexpr char thresh_output_msg[] = "Probability threshold for detections. Default is 0.5";
DEFINE_double(t, 0.5, thresh_output_msg);

constexpr char bb_enlarge_coef_output_msg[] = "Coefficient to enlarge/reduce the size of the bounding box around the detected face. Default is 1.2";
DEFINE_double(bb_enlarge_coef, 1.2, bb_enlarge_coef_output_msg);

constexpr char raw_output_msg[] = "Output inference results as raw values";
DEFINE_bool(r, false, raw_output_msg);

constexpr char no_show_msg[] = "Don't show output";
DEFINE_bool(no_show, false, no_show_msg);

constexpr char dx_coef_output_msg[] = "Coefficient to shift the bounding box around the detected face along the Ox axis";
DEFINE_double(dx_coef, 1, dx_coef_output_msg);

constexpr char dy_coef_output_msg[] = "Coefficient to shift the bounding box around the detected face along the Oy axis";
DEFINE_double(dy_coef, 1, dy_coef_output_msg);

constexpr char fps_output_msg[] = "Maximum FPS for playing video";
DEFINE_double(fps, -std::numeric_limits<double>::infinity(), fps_output_msg);

constexpr char no_smooth_output_msg[] = "Do not smooth person attributes";
DEFINE_bool(no_smooth, false, no_smooth_output_msg);

constexpr char no_show_emotion_bar_msg[] = "Do not show emotion bar";
DEFINE_bool(no_show_emotion_bar, false, no_show_emotion_bar_msg);

constexpr char utilization_monitors_msg[] = "List of monitors to show initially";
DEFINE_string(u, "", utilization_monitors_msg);

void parse(int argc, char *argv[]) {
    // ---------------------------Parsing and validating input arguments--------------------------------------
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    if (FLAGS_h || 1 == argc) {    
        std::cout << "  \t-h                         " << help_msg
                << "\n\t-i                         " << input_msg
                << "\n\t-loop                      " << loop_msg
                << "\n\t-o \"<path>\"              " << output_msg
                << "\n\t-limit \"<num>\"           " << limit_msg
                << "\n\t-m \"<path>\"              " << face_detection_model_msg
                << "\n\t[-m_ag] \"<path>\"         " << age_gender_model_msg
                << "\n\t[-m_hp] \"<path>\"         " << head_pose_model_msg
                << "\n\t[-m_em] \"<path>\"         " << emotions_model_msg
                << "\n\t[-m_lm] \"<path>\"         " << facial_landmarks_model_msg
                << "\n\t[-m_am] \"<path>\"         " << antispoofing_model_msg
                << "\n\t-d <device>                " << device_msg
                << "\n\t[-no_show]                 " << no_show_msg
                << "\n\t[-r]                       " << raw_output_msg
                << "\n\t[-t]                       " << thresh_output_msg
                << "\n\t[-bb_enlarge_coef]         " << bb_enlarge_coef_output_msg
                << "\n\t[-dx_coef]                 " << dx_coef_output_msg
                << "\n\t[-dy_coef]                 " << dy_coef_output_msg
                << "\n\t[-fps]                     " << fps_output_msg
                << "\n\t[-no_smooth]               " << no_smooth_output_msg
                << "\n\t[-no_show_emotion_bar]     " << no_show_emotion_bar_msg
                << "\n\t[-u]                       " << utilization_monitors_msg << '\n';
        showAvailableDevices();
        slog::info << ov::get_openvino_version() << slog::endl;
        exit(0);
    } if (FLAGS_i.empty()) {
        throw std::invalid_argument{"-i <INPUT> can't be empty"};
    } if (FLAGS_m.empty()) {
        throw std::invalid_argument{"-m <MODEL FILE> can't be empty"};
    }
    slog::info << ov::get_openvino_version() << slog::endl;
}
} // namespace

int main(int argc, char *argv[]) {
    PerformanceMetrics metrics;
    std::set_terminate(catcher);
    parse(argc, argv);

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

    cv::VideoWriter videoWriter;
    if (!FLAGS_o.empty() && !videoWriter.open(FLAGS_o, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                                !FLAGS_no_show && FLAGS_fps > 0.0 ? FLAGS_fps : cap->fps(),
                                                frame.size())) {
        throw std::runtime_error("Can't open video writer");
    }

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

        if (videoWriter.isOpened() && (FLAGS_limit == 0 || framesCounter <= FLAGS_limit)) {
            videoWriter.write(prevFrame);
        }

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

    return 0;
}
