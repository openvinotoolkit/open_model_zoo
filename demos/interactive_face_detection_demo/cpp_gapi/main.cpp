// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief The entry point for the G-API/Inference Engine interactive_face_detection_gapi demo application
* \file interactive_face_detection_demo_gapi/main.cpp
* \example interactive_face_detection_demo_gapi/main.cpp
*/
#include <vector>
#include <string>
#include <list>
#include <memory>

#include <utils/ocv_common.hpp>
#include <utils/slog.hpp>

#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/streaming/cap.hpp>

#include <monitors/presenter.h>

#include "interactive_face_detection_gapi.hpp"
#include "utils.hpp"
#include "face.hpp"
#include "visualizer.hpp"

static const std::vector<std::string> EMOTION_VECTOR = {"neutral",
                                                        "happy",
                                                        "sad",
                                                        "surprise",
                                                        "anger"};

using AGInfo = std::tuple<cv::GMat, cv::GMat>;
using HPInfo = std::tuple<cv::GMat, cv::GMat, cv::GMat>;
G_API_NET(Faces,          <cv::GMat(cv::GMat)>, "face-detector");
G_API_NET(AgeGender,      <AGInfo(cv::GMat)>,   "age-gender-recognition");
G_API_NET(HeadPose,       <HPInfo(cv::GMat)>,   "head-pose-recognition");
G_API_NET(FacialLandmark, <cv::GMat(cv::GMat)>, "facial-landmark-recognition");
G_API_NET(Emotions,       <cv::GMat(cv::GMat)>, "emotions-recognition");

G_API_OP(PostProc, <cv::GArray<cv::Rect>(cv::GMat, cv::GMat, double, double, double, double)>, "custom.fd_postproc") {
    static cv::GArrayDesc outMeta(const cv::GMatDesc &, const cv::GMatDesc &, double, double, double, double) {
        return cv::empty_array_desc();
    }
};

GAPI_OCV_KERNEL(OCVPostProc, PostProc) {
    static void run(const cv::Mat &in_ssd_result,
                    const cv::Mat &in_frame,
                    double threshold,
                    double bb_enlarge_coefficient,
                    double bb_dx_coefficient,
                    double bb_dy_coefficient,
                    std::vector<cv::Rect> &out_faces) {
        const auto &in_ssd_dims = in_ssd_result.size;
        CV_Assert(in_ssd_dims.dims() == 4u);

        const int MAX_PROPOSALS = in_ssd_dims[2];
        const int OBJECT_SIZE   = in_ssd_dims[3];
        CV_Assert(OBJECT_SIZE == 7);

        const cv::Size upscale = in_frame.size();
        const cv::Rect surface({0,0}, upscale);
        out_faces.clear();

        const float *data = in_ssd_result.ptr<float>();
        for (int i = 0; i < MAX_PROPOSALS; i++) {
            const float image_id   = data[i * OBJECT_SIZE + 0]; // batch id
            const float confidence = data[i * OBJECT_SIZE + 2];
            const float rc_left    = data[i * OBJECT_SIZE + 3];
            const float rc_top     = data[i * OBJECT_SIZE + 4];
            const float rc_right   = data[i * OBJECT_SIZE + 5];
            const float rc_bottom  = data[i * OBJECT_SIZE + 6];

            if (image_id < 0.f) {  // indicates end of detections
                break;
            }
            if (confidence < threshold) {
                continue;
            }

            cv::Rect rc;
            rc.x      = static_cast<int>(rc_left   * upscale.width);
            rc.y      = static_cast<int>(rc_top    * upscale.height);
            rc.width  = static_cast<int>(rc_right  * upscale.width)  - rc.x;
            rc.height = static_cast<int>(rc_bottom * upscale.height) - rc.y;

            // Make square and enlarge face bounding box for more robust operation of face analytics networks
            int bb_width = rc.width;
            int bb_height = rc.height;

            int bb_center_x = rc.x + bb_width / 2;
            int bb_center_y = rc.y + bb_height / 2;

            int max_of_sizes = std::max(bb_width, bb_height);

            int bb_new_width = static_cast<int>(bb_enlarge_coefficient * max_of_sizes);
            int bb_new_height = static_cast<int>(bb_enlarge_coefficient * max_of_sizes);

            rc.x = bb_center_x - static_cast<int>(std::floor(bb_dx_coefficient * bb_new_width / 2));
            rc.y = bb_center_y - static_cast<int>(std::floor(bb_dy_coefficient * bb_new_height / 2));

            rc.width = bb_new_width;
            rc.height = bb_new_height;

            out_faces.push_back(rc & surface);
        }
    }
};

void rawOutputDetections(const cv::Mat  &ssd_result,
                         const cv::Size &upscale,
                         const double detectionThreshold) {
    const auto &in_ssd_dims = ssd_result.size;
    CV_Assert(in_ssd_dims.dims() == 4u);

    const int OBJECT_SIZE   = in_ssd_dims[3];
    CV_Assert(OBJECT_SIZE == 7);

    const float *data = ssd_result.ptr<float>();

    const int detection_num = in_ssd_dims[2];
    for (int i = 0; i < detection_num; ++i) {
        const float image_id   = data[i * OBJECT_SIZE + 0];
        const float label      = data[i * OBJECT_SIZE + 1];
        const float confidence = data[i * OBJECT_SIZE + 2];
        const float rc_left    = data[i * OBJECT_SIZE + 3];
        const float rc_top     = data[i * OBJECT_SIZE + 4];
        const float rc_right   = data[i * OBJECT_SIZE + 5];
        const float rc_bottom  = data[i * OBJECT_SIZE + 6];

        if (image_id < 0.f) {  // indicates end of detections
            break;
        }

        int x      = static_cast<int>(rc_left   * upscale.width);
        int y      = static_cast<int>(rc_top    * upscale.height);
        int width  = static_cast<int>(rc_right  * upscale.width)  - x;
        int height = static_cast<int>(rc_bottom * upscale.height) - y;

        std::cout << "[" << i << "," << label << "] element, prob = " << confidence <<
             "    (" << x << "," << y << ")-(" << width << "," << height << ")"
             << ((confidence > detectionThreshold) ? " WILL BE RENDERED!" : "") << std::endl;
    }
}

void rawOutputAgeGender(const int idx, const cv::Mat &out_ages, const cv::Mat &out_genders) {
    const float *age_data = out_ages.ptr<float>();
    const float *gender_data = out_genders.ptr<float>();

    float maleProb = gender_data[1];
    float age      = age_data[0] * 100;

    std::cout << "[" << idx << "] element, male prob = " << maleProb << ", age = " << age << std::endl;
}

void rawOutputHeadpose(const int idx,
                       const cv::Mat &out_y_fc,
                       const cv::Mat &out_p_fc,
                       const cv::Mat &out_r_fc) {
    const float *y_data = out_y_fc.ptr<float>();
    const float *p_data = out_p_fc.ptr<float>();
    const float *r_data = out_r_fc.ptr<float>();

    std::cout << "[" << idx << "] element, yaw = " << y_data[0] <<
                 ", pitch = " << p_data[0] <<
                 ", roll = " << r_data[0]  << std::endl;
}

void rawOutputLandmarks(const int idx, const cv::Mat &out_landmark) {
    const float *lm_data = out_landmark.ptr<float>();

    std::cout << "[" << idx << "] element, normed facial landmarks coordinates (x, y):" << std::endl;

    int n_lm = 70;
    for (int i_lm = 0; i_lm < n_lm / 2; ++i_lm) {
        float normed_x = lm_data[2 * i_lm];
        float normed_y = lm_data[2 * i_lm + 1];

        std::cout << normed_x << ", " << normed_y << std::endl;
    }
}

void rawOutputEmotions(const int idx, const cv::Mat &out_emotion) {
    size_t emotionsVecSize = EMOTION_VECTOR.size();

    const float *em_data = out_emotion.ptr<float>();

    std::cout << "[" << idx << "] element, predicted emotions (name = prob):" << std::endl;
    for (size_t i = 0; i < emotionsVecSize; i++) {
        std::cout << EMOTION_VECTOR[i] << " = " << em_data[i];
        if (emotionsVecSize - 1 != i) {
            std::cout << ", ";
        } else {
            std::cout << std::endl;
        }
    }
}

float calcMean(const cv::Mat& src) {
    cv::Mat tmp;
    cv::cvtColor(src, tmp, cv::COLOR_BGR2GRAY);
    cv::Scalar mean = cv::mean(tmp);

    return static_cast<float>(mean[0]);
}

void faceDataUpdate(const cv::Mat &frame,
                    Face::Ptr &face,
                    const cv::Rect &face_rect,
                    std::list<Face::Ptr>  &prev_faces,
                    const std::vector<cv::Rect> &face_hub,
                    size_t &id,
                    bool no_smooth) {
    // Face update
    cv::Rect rect = face_rect & cv::Rect({0, 0}, frame.size());

    if (!no_smooth) {
        face = matchFace(rect, prev_faces);
        float intensity_mean = calcMean(frame(rect));
        intensity_mean += 1.0;

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
}

void ageGenderDataUpdate(const Face::Ptr &face,
                         const cv::Mat &out_age,
                         const cv::Mat &out_gender) {
    const float *age_data =    out_age.ptr<float>();
    const float *gender_data = out_gender.ptr<float>();

    float maleProb = gender_data[1];
    float age      = age_data[0] * 100;

    face->updateGender(maleProb);
    face->updateAge(age);
}

void headPoseDataUpdate(const Face::Ptr &face,
                        const cv::Mat &out_y_fc,
                        const cv::Mat &out_p_fc,
                        const cv::Mat &out_r_fc) {
    const float *y_data = out_y_fc.ptr<float>();
    const float *p_data = out_p_fc.ptr<float>();
    const float *r_data = out_r_fc.ptr<float>();

    face->updateHeadPose(y_data[0], p_data[0], r_data[0]);
}

void emotionsDataUpdate(const Face::Ptr &face, const cv::Mat &out_emotion) {
    const float *em_data = out_emotion.ptr<float>();

    std::map<std::string, float> em_val_map;
    for(size_t i = 0; i  < EMOTION_VECTOR.size(); i++) {
        em_val_map[EMOTION_VECTOR[i]] = em_data[i];
    }

    face->updateEmotions(em_val_map);
}

void landmarksDataUpdate(const Face::Ptr &face, const cv::Mat &out_landmark) {
    const float *lm_data = out_landmark.ptr<float>();

    size_t n_lm = 70;

    std::vector<float> normedLandmarks(&lm_data[0], &lm_data[n_lm]);

    face->updateLandmarks(normedLandmarks);
}

void setInput(cv::GStreamingCompiled stream, const std::string& input ) {
    try {
        // If stoi() throws exception input should be a path not a camera id
        stream.setSource(cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(std::stoi(input)));
    } catch (std::invalid_argument&) {
        slog::info << "Input source is treated as a file path" << slog::endl;
        stream.setSource(cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(input));
    }
}

int main(int argc, char *argv[]) {
    try {
        // ------------------------------ Parsing and validating of input arguments --------------------------

        slog::info << "Parsing input parameters" << slog::endl;
        gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
        if (FLAGS_h) {
            showUsage();
            showAvailableDevices();
            return 0;
        }

        if (FLAGS_i.empty())
            throw std::logic_error("Parameter -i is not set");
        if (FLAGS_m.empty())
            throw std::logic_error("Parameter -m is not set");

        std::cout << "To close the application, press 'CTRL+C' here";
        if (!FLAGS_no_show) {
            std::cout << " or switch to the output window and press any key";
        }
        std::cout << std::endl;

        cv::GComputation pipeline([=]() {
                cv::GMat in;

                cv::GMat frame = cv::gapi::copy(in);

                cv::GMat detections = cv::gapi::infer<Faces>(in);

                cv::GArray<cv::Rect> faces = PostProc::on(detections, in,
                                                          FLAGS_t,
                                                          FLAGS_bb_enlarge_coef,
                                                          FLAGS_dx_coef,
                                                          FLAGS_dy_coef);
                auto outs = GOut(frame, detections, faces);

                cv::GArray<cv::GMat> ages, genders;
                if (!FLAGS_m_ag.empty()) {
                    std::tie(ages, genders) = cv::gapi::infer<AgeGender>(faces, in);
                    outs += GOut(ages, genders);
                }

                cv::GArray<cv::GMat> y_fc, p_fc, r_fc;
                if (!FLAGS_m_hp.empty()) {
                    std::tie(y_fc, p_fc, r_fc) = cv::gapi::infer<HeadPose>(faces, in);
                    outs += GOut(y_fc, p_fc, r_fc);
                }

                cv::GArray<cv::GMat> emotions;
                if (!FLAGS_m_em.empty()) {
                    emotions = cv::gapi::infer<Emotions>(faces, in);
                    outs += GOut(emotions);
                }

                cv::GArray<cv::GMat> landmarks;
                if (!FLAGS_m_lm.empty()) {
                    landmarks = cv::gapi::infer<FacialLandmark>(faces, in);
                    outs += GOut(landmarks);
                }

                return cv::GComputation(cv::GIn(in), std::move(outs));
        });

        auto det_net = cv::gapi::ie::Params<Faces> {
            FLAGS_m,                         // path to model
            fileNameNoExt(FLAGS_m) + ".bin", // path to weights
            FLAGS_d                          // device to use
        };

        auto age_net = cv::gapi::ie::Params<AgeGender> {
            FLAGS_m_ag,                         // path to model
            fileNameNoExt(FLAGS_m_ag) + ".bin", // path to weights
            FLAGS_d_ag                          // device to use
        }.cfgOutputLayers({ "age_conv3", "prob" });


        auto hp_net = cv::gapi::ie::Params<HeadPose> {
            FLAGS_m_hp,                         // path to model
            fileNameNoExt(FLAGS_m_hp) + ".bin", // path to weights
            FLAGS_d_hp                          // device to use
        }.cfgOutputLayers({ "angle_y_fc", "angle_p_fc", "angle_r_fc" });

        auto lm_net = cv::gapi::ie::Params<FacialLandmark> {
            FLAGS_m_lm,                        // path to model
            fileNameNoExt(FLAGS_m_lm) + ".bin",// path to weights
            FLAGS_d_lm                         // device to use
        }.cfgOutputLayers({ "align_fc3" });

        auto emo_net = cv::gapi::ie::Params<Emotions> {
            FLAGS_m_em,                         // path to model
            fileNameNoExt(FLAGS_m_em) + ".bin", // path to weights
            FLAGS_d_em                          // device to use
        };

        // Form a kernel package (including an OpenCV-based implementation of our
        // post-processing) and a network package (holding our three networks).
        auto kernels = cv::gapi::kernels<OCVPostProc>();
        auto networks = cv::gapi::networks(det_net, age_net, hp_net, lm_net, emo_net);

        cv::GStreamingCompiled stream = pipeline.compileStreaming(cv::compile_args(kernels, networks));

        cv::Mat frame, ssd_res;
        std::vector<cv::Rect> face_hub;
        auto out_vector = cv::gout(frame, ssd_res, face_hub);

        std::vector<cv::Mat> out_ages, out_genders;
        if (!FLAGS_m_ag.empty()) out_vector += cv::gout(out_ages, out_genders);

        std::vector<cv::Mat> out_y_fc, out_p_fc, out_r_fc;
        if (!FLAGS_m_hp.empty()) out_vector += cv::gout(out_y_fc, out_p_fc, out_r_fc);

        std::vector<cv::Mat> out_emotions;
        if (!FLAGS_m_em.empty()) out_vector += cv::gout(out_emotions);

        std::vector<cv::Mat> out_landmarks;
        if (!FLAGS_m_lm.empty()) out_vector += cv::gout(out_landmarks);

        Visualizer::Ptr visualizer;
        if (!FLAGS_no_show || !FLAGS_o.empty()) {
            visualizer = std::make_shared<Visualizer>(!FLAGS_m_ag.empty(), !FLAGS_m_em.empty(), !FLAGS_m_hp.empty(), !FLAGS_m_lm.empty());
        } else {
            std::cout<< "To close the application, press 'CTRL+C' here" << std::endl;
        }

        std::list<Face::Ptr> faces;
        std::ostringstream out;
        size_t framesCounter = 0;
        size_t id = 0;
        cv::VideoWriter videoWriter;

        const cv::Point THROUGHPUT_METRIC_POSITION{10, 45};
        std::unique_ptr<Presenter> presenter;

        Timer timer;
        do {
            slog::info << "Setting media source" << slog::endl;
            try {
                setInput(stream, FLAGS_i);
            } catch (const std::exception& error) {
                std::stringstream msg;
                msg << "Can't open source {" << FLAGS_i << "}" << std::endl <<
                    error.what() << std::endl;
                throw std::invalid_argument(msg.str());
            }
            slog::info << "Start inference " << slog::endl;

            timer.start("total");
            stream.start();
            while (stream.pull(cv::GRunArgsP(out_vector))) {
                if (!FLAGS_no_show && !FLAGS_m_em.empty() && !FLAGS_no_show_emotion_bar) {
                    visualizer->enableEmotionBar(frame.size(), EMOTION_VECTOR);
                }

                // Init presenter
                if (presenter == nullptr) {
                    cv::Size graphSize{static_cast<int>(frame.rows / 4), 60};
                    presenter.reset(new Presenter(FLAGS_u, THROUGHPUT_METRIC_POSITION.y + 15, graphSize));
                }

                //  Postprocessing
                std::list<Face::Ptr> prev_faces;

                if (!FLAGS_no_smooth) {
                    prev_faces.insert(prev_faces.begin(), faces.begin(), faces.end());
                }

                faces.clear();

                // Raw output of detected faces
                if (FLAGS_r)
                    rawOutputDetections(ssd_res, frame.size(), FLAGS_t);

                // For every detected face
                for (size_t i = 0; i < face_hub.size(); i++) {
                    Face::Ptr face;

                    cv::Rect rect = face_hub[i] & cv::Rect({0, 0}, frame.size());
                    faceDataUpdate(frame, face, rect,
                                   prev_faces, face_hub,
                                   id, FLAGS_no_smooth);

                    if (!FLAGS_m_ag.empty()) {
                        ageGenderDataUpdate(face, out_ages[i], out_genders[i]);
                        if (FLAGS_r)
                            rawOutputAgeGender(i, out_ages[i], out_genders[i]);
                    }

                    if (!FLAGS_m_em.empty()) {
                        emotionsDataUpdate(face, out_emotions[i]);
                        if (FLAGS_r)
                            rawOutputEmotions(i, out_emotions[i]);
                    }

                    if (!FLAGS_m_hp.empty()) {
                        headPoseDataUpdate(face, out_y_fc[i], out_p_fc[i], out_r_fc[i]);
                        if (FLAGS_r)
                            rawOutputHeadpose(i, out_y_fc[i], out_p_fc[i], out_r_fc[i]);
                    }

                    if (!FLAGS_m_lm.empty()) {
                        landmarksDataUpdate(face, out_landmarks[i]);
                        if (FLAGS_r)
                            rawOutputLandmarks(i, out_landmarks[i]);
                    }
                    // End of face postprocessing

                    faces.push_back(face);
                }

                presenter->drawGraphs(frame);

                //  Visualizing results
                if (!FLAGS_no_show || !FLAGS_o.empty()) {
                    out.str("");
                    out << "Total image throughput: " << std::fixed << std::setprecision(2)
                        << 1000.f / (timer["total"].getSmoothedDuration()) << " fps";
                    cv::putText(frame, out.str(), THROUGHPUT_METRIC_POSITION, cv::FONT_HERSHEY_TRIPLEX, 1,
                                cv::Scalar(255, 0, 0), 2);

                    // drawing faces
                    visualizer->draw(frame, faces);

                    cv::imshow("Detection results", frame);

                    int key = cv::waitKey(1);
                    if (27 == key || 'Q' == key || 'q' == key) {
                        stream.stop();
                    } else {
                        presenter->handleKey(key);
                    }
                }
                if (!FLAGS_o.empty() && framesCounter == 0 &&
                    !videoWriter.open(FLAGS_o, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 25, frame.size())) {
                    throw std::runtime_error("Can't open video writer");
                }
                if (videoWriter.isOpened() && (FLAGS_limit == 0 || framesCounter <= FLAGS_limit - 1)) {
                    videoWriter.write(frame);
                }

                timer["total"].calculateDuration();
                framesCounter++;
            }
            timer.finish("total");

            slog::info << "Number of processed frames: " << framesCounter << slog::endl;
            slog::info << "Total image throughput: " << framesCounter * (1000.f / timer["total"].getTotalDuration()) << " fps" << slog::endl;

            std::cout << presenter->reportMeans() << '\n';
        } while (FLAGS_loop);

        slog::info << "No more frames to process!" << slog::endl;

        cv::destroyAllWindows();
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
