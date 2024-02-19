// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include <limits>

#include <gflags/gflags.h>
#include <utils/images_capture.h>
#include <monitors/presenter.h>

#include "detectors.hpp"
#include "face.hpp"
#include "visualizer.hpp"

namespace {
constexpr char h_msg[] = "show the help message and exit";
DEFINE_bool(h, false, h_msg);

constexpr char m_msg[] = "path to an .xml file with a trained Face Detection model";
DEFINE_string(m, "", m_msg);

constexpr char i_msg[] = "an input to process. The input must be a single image, a folder of images, video file or camera id. Default is 0";
DEFINE_string(i, "0", i_msg);

constexpr char bb_enlarge_coef_msg[] = "coefficient to enlarge/reduce the size of the bounding box around the detected face. Default is 1.2";
DEFINE_double(bb_enlarge_coef, 1.2, bb_enlarge_coef_msg);

constexpr char d_msg[] =
    "specify a device to infer on (the list of available devices is shown below). "
    "Use '-d HETERO:<comma-separated_devices_list>' format to specify HETERO plugin. "
    "Use '-d MULTI:<comma-separated_devices_list>' format to specify MULTI plugin. "
    "Default is CPU";
DEFINE_string(d, "CPU", d_msg);

constexpr char dx_coef_msg[] = "coefficient to shift the bounding box around the detected face along the Ox axis";
DEFINE_double(dx_coef, 1, dx_coef_msg);

constexpr char dy_coef_msg[] = "coefficient to shift the bounding box around the detected face along the Oy axis";
DEFINE_double(dy_coef, 1, dy_coef_msg);

constexpr char fps_msg[] = "maximum FPS for playing video";
DEFINE_double(fps, -std::numeric_limits<double>::infinity(), fps_msg);

constexpr char lim_msg[] = "number of frames to store in output. If 0 is set, all frames are stored. Default is 1000";
DEFINE_uint32(lim, 1000, lim_msg);

constexpr char loop_msg[] = "enable reading the input in a loop";
DEFINE_bool(loop, false, loop_msg);

constexpr char mag_msg[] = "path to an .xml file with a trained Age/Gender Recognition model";
DEFINE_string(mag, "", mag_msg);

constexpr char mam_msg[] = "path to an .xml file with a trained Antispoofing Classification model";
DEFINE_string(mam, "", mam_msg);

constexpr char mem_msg[] = "path to an .xml file with a trained Emotions Recognition model";
DEFINE_string(mem, "", mem_msg);

constexpr char mhp_msg[] = "path to an .xml file with a trained Head Pose Estimation model";
DEFINE_string(mhp, "", mhp_msg);

constexpr char mlm_msg[] = "path to an .xml file with a trained Facial Landmarks Estimation model";
DEFINE_string(mlm, "", mlm_msg);

constexpr char o_msg[] = "name of the output file(s) to save";
DEFINE_string(o, "", o_msg);

constexpr char r_msg[] = "output inference results as raw values";
DEFINE_bool(r, false, r_msg);

constexpr char show_msg[] = "(don't) show output";
DEFINE_bool(show, true, show_msg);

constexpr char show_emotion_bar_msg[] = "(don't) show emotion bar";
DEFINE_bool(show_emotion_bar, true, show_emotion_bar_msg);

constexpr char smooth_msg[] = "(don't) smooth person attributes";
DEFINE_bool(smooth, true, smooth_msg);

constexpr char t_msg[] = "probability threshold for detections. Default is 0.5";
DEFINE_double(t, 0.5, t_msg);

constexpr char u_msg[] = "resource utilization graphs. Default is cdm. "
    "c - average CPU load, d - load distribution over cores, m - memory usage, h - hide";
DEFINE_string(u, "cdm", u_msg);

void parse(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    if (FLAGS_h || 1 == argc) {
        std::cout <<   "\t[ -h]                                         " << h_msg
                  << "\n\t[--help]                                      print help on all arguments"
                  << "\n\t  -m <MODEL FILE>                             " << m_msg
                  << "\n\t[ -i <INPUT>]                                 " << i_msg
                  << "\n\t[--bb_enlarge_coef <NUMBER>]                  " << bb_enlarge_coef_msg
                  << "\n\t[ -d <DEVICE>]                                " << d_msg
                  << "\n\t[--dx_coef <NUMBER>]                          " << dx_coef_msg
                  << "\n\t[--dy_coef <NUMBER>]                          " << dy_coef_msg
                  << "\n\t[--fps <NUMBER>]                              " << fps_msg
                  << "\n\t[--lim <NUMBER>]                              " << lim_msg
                  << "\n\t[--loop]                                      " << loop_msg
                  << "\n\t[--mag <MODEL FILE>]                          " << mag_msg
                  << "\n\t[--mam <MODEL FILE>]                          " << mam_msg
                  << "\n\t[--mem <MODEL FILE>]                          " << mem_msg
                  << "\n\t[--mhp <MODEL FILE>]                          " << mhp_msg
                  << "\n\t[--mlm <MODEL FILE>]                          " << mlm_msg
                  << "\n\t[ -o <OUTPUT>]                                " << o_msg
                  << "\n\t[ -r]                                         " << r_msg
                  << "\n\t[--show] ([--noshow])                         " << show_msg
                  << "\n\t[--show_emotion_bar] ([--noshow_emotion_bar]) " << show_emotion_bar_msg
                  << "\n\t[--smooth] ([--nosmooth])                     " << smooth_msg
                  << "\n\t[ -t <NUMBER>]                                " << t_msg
                  << "\n\t[ -u <DEVICE>]                                " << u_msg
                  << "\n\tKey bindings:"
                     "\n\t\tQ, q, Esc - Quit"
                     "\n\t\tP, p, 0, spacebar - Pause"
                     "\n\t\tC - average CPU load, D - load distribution over cores, M - memory usage, H - hide\n";
        showAvailableDevices();
        std::cout << ov::get_openvino_version() << std::endl;
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
    std::set_terminate(catcher);
    parse(argc, argv);
    PerformanceMetrics metrics;

    // --------------------------- 1. Loading Inference Engine -----------------------------
    ov::Core core;

    FaceDetection faceDetector(FLAGS_m, FLAGS_t, FLAGS_r,
                                static_cast<float>(FLAGS_bb_enlarge_coef), static_cast<float>(FLAGS_dx_coef), static_cast<float>(FLAGS_dy_coef));
    AgeGenderDetection ageGenderDetector(FLAGS_mag, FLAGS_r);
    HeadPoseDetection headPoseDetector(FLAGS_mhp, FLAGS_r);
    EmotionsDetection emotionsDetector(FLAGS_mem, FLAGS_r);
    FacialLandmarksDetection facialLandmarksDetector(FLAGS_mlm, FLAGS_r);
    AntispoofingClassifier antispoofingClassifier(FLAGS_mam, FLAGS_r);
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
    if (FLAGS_show_emotion_bar && emotionsDetector.enabled()) {
        visualizer.enableEmotionBar(emotionsDetector.emotionsVec);
    }

    LazyVideoWriter videoWriter{FLAGS_o, FLAGS_fps > 0.0 ? FLAGS_fps : cap->fps(), FLAGS_lim};

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

        if (!FLAGS_smooth) {
            prev_faces.insert(prev_faces.begin(), faces.begin(), faces.end());
        }

        faces.clear();

        // For every detected face
        for (size_t i = 0; i < prev_detection_results.size(); i++) {
            auto& result = prev_detection_results[i];
            cv::Rect rect = result.location & cv::Rect({0, 0}, prevFrame.size());

            Face::Ptr face;
            if (FLAGS_smooth) {
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
        if (FLAGS_show) {
            cv::imshow(argv[0], prevFrame);
            int key = cv::waitKey(delay);
            if ('P' == key || 'p' == key || '0' == key || ' ' == key) {
                key = cv::waitKey(0);
            }
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
