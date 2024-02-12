// Copyright (C) 2020-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief The entry point for the G-API interactive_face_detection_gapi demo application
 * \file interactive_face_detection_demo_gapi/main.cpp
 * \example interactive_face_detection_demo_gapi/main.cpp
 */
#include <stdlib.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gflags/gflags.h>
#include <opencv2/core.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/garray.hpp>
#include <opencv2/gapi/gcommon.hpp>
#include <opencv2/gapi/gcomputation.hpp>
#include <opencv2/gapi/gkernel.hpp>
#include <opencv2/gapi/gmat.hpp>
#include <opencv2/gapi/gopaque.hpp>
#include <opencv2/gapi/gproto.hpp>
#include <opencv2/gapi/gstreaming.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/infer/ov.hpp>
#include <opencv2/gapi/infer/parsers.hpp>
#include <opencv2/gapi/streaming/format.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

#include <monitors/presenter.h>
#include <utils/common.hpp>
#include <utils/images_capture.h>
#include <utils/ocv_common.hpp>
#include <utils/performance_metrics.hpp>
#include <utils/slog.hpp>
#include <utils_gapi/stream_source.hpp>

#include "face.hpp"
#include "visualizer.hpp"

namespace {
constexpr char h_msg[] = "show the help message and exit";
DEFINE_bool(h, false, h_msg);

constexpr char m_msg[] = "path to an .xml file with a trained Face Detection model";
DEFINE_string(m, "", m_msg);

constexpr char i_msg[] =
    "an input to process. The input must be a single image, a folder of images, video file or camera id. Default is 0";
DEFINE_string(i, "0", i_msg);

constexpr char bb_enlarge_coef_msg[] =
    "coefficient to enlarge/reduce the size of the bounding box around the detected face. Default is 1.2";
DEFINE_double(bb_enlarge_coef, 1.2, bb_enlarge_coef_msg);

constexpr char d_msg[] = "target device for Face Detection network (the list of available devices is shown below). "
                         "The demo will look for a suitable plugin for a specified device. Default is CPU";
DEFINE_string(d, "CPU", d_msg);

constexpr char dag_msg[] =
    "target device for Age/Gender Recognition network (the list of available devices is shown below). "
    "The demo will look for a suitable plugin for a specified device. Default is CPU";
DEFINE_string(dag, "CPU", dag_msg);

constexpr char dam_msg[] =
    "target device for Antispoofing Classification network (the list of available devices is shown below). "
    "Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. "
    "The demo will look for a suitable plugin for a specified device. Default is CPU";
DEFINE_string(dam, "CPU", dam_msg);

constexpr char dem_msg[] =
    "target device for Emotions Recognition network (the list of available devices is shown below). "
    "The demo will look for a suitable plugin for a specified device. Default is CPU";
DEFINE_string(dem, "CPU", dem_msg);

constexpr char dhp_msg[] =
    "target device for Head Pose Estimation network (the list of available devices is shown below). "
    "The demo will look for a suitable plugin for a specified device. Default is CPU";
DEFINE_string(dhp, "CPU", dhp_msg);

constexpr char dlm_msg[] = "target device for Facial Landmarks Estimation network "
                           "(the list of available devices is shown below). The demo will look for a suitable plugin "
                           "for device specified. Default is CPU";
DEFINE_string(dlm, "CPU", dlm_msg);

constexpr char dx_coef_msg[] = "coefficient to shift the bounding box around the detected face along the Ox axis";
DEFINE_double(dx_coef, 1, dx_coef_msg);

constexpr char dy_coef_msg[] = "coefficient to shift the bounding box around the detected face along the Oy axis";
DEFINE_double(dy_coef, 1, dy_coef_msg);

constexpr char lim_msg[] = "number of frames to store in output. If 0 is set, all frames are stored. Default is 1000";
DEFINE_uint32(lim, 1000, lim_msg);

// TODO: Make this option valid for single image case
constexpr char loop_msg[] = "enable playing video on a loop";
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

void parse(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    if (FLAGS_h || 1 == argc) {
        std::cout << "\t[ -h]                                         " << h_msg
                  << "\n\t[--help]                                      print help on all arguments"
                  << "\n\t  -m <MODEL FILE>                             " << m_msg
                  << "\n\t[ -i <INPUT>]                                 " << i_msg
                  << "\n\t[--bb_enlarge_coef <NUMBER>]                  " << bb_enlarge_coef_msg
                  << "\n\t[ -d <DEVICE>]                                " << d_msg
                  << "\n\t[--dag <DEVICE>]                              " << dag_msg
                  << "\n\t[--dam <DEVICE>]                              " << dam_msg
                  << "\n\t[--dem <DEVICE>]                              " << dem_msg
                  << "\n\t[--dhp <DEVICE>]                              " << dhp_msg
                  << "\n\t[--dlm <DEVICE>]                              " << dlm_msg
                  << "\n\t[--dx_coef <NUMBER>]                          " << dx_coef_msg
                  << "\n\t[--dy_coef <NUMBER>]                          " << dy_coef_msg
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
    }
    if (FLAGS_i.empty()) {
        throw std::invalid_argument{"-i <INPUT> can't be empty"};
    }
    if (FLAGS_m.empty()) {
        throw std::invalid_argument{"-m <MODEL FILE> can't be empty"};
    }
    slog::info << ov::get_openvino_version() << slog::endl;
}

static const std::vector<std::string> EMOTION_VECTOR = {"neutral", "happy", "sad", "surprise", "anger"};

using AGInfo = std::tuple<cv::GMat, cv::GMat>;
using HPInfo = std::tuple<cv::GMat, cv::GMat, cv::GMat>;
G_API_NET(Faces, <cv::GMat(cv::GMat)>, "face-detector");
G_API_NET(AgeGender, <AGInfo(cv::GMat)>, "age-gender-recognition");
G_API_NET(HeadPose, <HPInfo(cv::GMat)>, "head-pose-recognition");
G_API_NET(FacialLandmark, <cv::GMat(cv::GMat)>, "facial-landmark-recognition");
G_API_NET(Emotions, <cv::GMat(cv::GMat)>, "emotions-recognition");
G_API_NET(ASpoof, <cv::GMat(cv::GMat)>, "anti-spoofing");

// clang-format off
G_API_OP(PostProc, <cv::GArray<cv::Rect>(cv::GArray<cv::Rect>,
    cv::GOpaque<cv::Size>, double, double, double)>, "custom.fd_postproc") {
    static cv::GArrayDesc outMeta(const cv::GArrayDesc&, const cv::GOpaqueDesc&, double, double, double) {
        return cv::empty_array_desc();
    }
};

GAPI_OCV_KERNEL(OCVPostProc, PostProc) {
    static void run(const std::vector<cv::Rect>& rois,
                    const cv::Size& frame_size,
                    double bb_enlarge_coefficient,
                    double bb_dx_coefficient,
                    double bb_dy_coefficient,
                    std::vector<cv::Rect> &out_faces) {
        out_faces.clear();
        const cv::Rect surface({0, 0}, frame_size);
        for (const auto& rc : rois) {
            // Make square and enlarge face bounding box for more robust operation of face analytics networks
            const int bb_width = rc.width;
            const int bb_height = rc.height;

            const int bb_center_x = rc.x + bb_width / 2;
            const int bb_center_y = rc.y + bb_height / 2;

            const int max_of_sizes = std::max(bb_width, bb_height);

            const int bb_new_width = static_cast<int>(bb_enlarge_coefficient * max_of_sizes);
            const int bb_new_height = static_cast<int>(bb_enlarge_coefficient * max_of_sizes);

            cv::Rect square_rect;
            square_rect.x = bb_center_x - static_cast<int>(std::floor(bb_dx_coefficient * bb_new_width / 2));
            square_rect.y = bb_center_y - static_cast<int>(std::floor(bb_dy_coefficient * bb_new_height / 2));

            square_rect.width = bb_new_width;
            square_rect.height = bb_new_height;

            out_faces.push_back(square_rect & surface);
        }
    }
};
// clang-format on

void rawOutputDetections(const cv::Mat& ssd_result, const cv::Size& upscale, const double detectionThreshold) {
    const auto& in_ssd_dims = ssd_result.size;
    CV_Assert(in_ssd_dims.dims() == 4u);

    const int OBJECT_SIZE = in_ssd_dims[3];
    CV_Assert(OBJECT_SIZE == 7);

    const float* data = ssd_result.ptr<float>();

    const int detection_num = in_ssd_dims[2];
    for (int i = 0; i < detection_num; ++i) {
        const float image_id = data[i * OBJECT_SIZE + 0];
        const float label = data[i * OBJECT_SIZE + 1];
        const float confidence = data[i * OBJECT_SIZE + 2];
        const float rc_left = data[i * OBJECT_SIZE + 3];
        const float rc_top = data[i * OBJECT_SIZE + 4];
        const float rc_right = data[i * OBJECT_SIZE + 5];
        const float rc_bottom = data[i * OBJECT_SIZE + 6];

        if (image_id < 0.f) {  // indicates end of detections
            break;
        }

        int x = static_cast<int>(rc_left * upscale.width);
        int y = static_cast<int>(rc_top * upscale.height);
        int width = static_cast<int>(rc_right * upscale.width) - x;
        int height = static_cast<int>(rc_bottom * upscale.height) - y;

        slog::debug << "[" << i << "," << label << "] element, prob = " << confidence << "    (" << x << "," << y
                    << ")-(" << width << "," << height << ")"
                    << ((confidence > detectionThreshold) ? " WILL BE RENDERED!" : "") << slog::endl;
    }
}

void rawOutputAgeGender(const int idx, const cv::Mat& out_ages, const cv::Mat& out_genders) {
    const float* age_data = out_ages.ptr<float>();
    const float* gender_data = out_genders.ptr<float>();

    const float maleProb = gender_data[1];
    const float age = age_data[0] * 100;

    slog::debug << "[" << idx << "] element, male prob = " << maleProb << ", age = " << age << slog::endl;
}

void rawOutputHeadpose(const int idx, const cv::Mat& out_y_fc, const cv::Mat& out_p_fc, const cv::Mat& out_r_fc) {
    const float* y_data = out_y_fc.ptr<float>();
    const float* p_data = out_p_fc.ptr<float>();
    const float* r_data = out_r_fc.ptr<float>();

    slog::debug << "[" << idx << "] element, yaw = " << y_data[0] << ", pitch = " << p_data[0]
                << ", roll = " << r_data[0] << slog::endl;
}

void rawOutputLandmarks(const int idx, const cv::Mat& out_landmark) {
    const float* lm_data = out_landmark.ptr<float>();

    slog::debug << "[" << idx << "] element, normed facial landmarks coordinates (x, y):" << slog::endl;

    int n_lm = 70;
    for (int i_lm = 0; i_lm < n_lm / 2; ++i_lm) {
        const float normed_x = lm_data[2 * i_lm];
        const float normed_y = lm_data[2 * i_lm + 1];

        slog::debug << '\t' << normed_x << ", " << normed_y << slog::endl;
    }
}

void rawOutputEmotions(const int idx, const cv::Mat& out_emotion) {
    const size_t emotionsVecSize = EMOTION_VECTOR.size();

    const float* em_data = out_emotion.ptr<float>();

    slog::debug << "[" << idx << "] element, predicted emotions (name = prob):" << slog::endl;
    for (size_t i = 0; i < emotionsVecSize; i++) {
        slog::debug << EMOTION_VECTOR[i] << " = " << em_data[i];
        if (emotionsVecSize - 1 != i) {
            slog::debug << ", ";
        } else {
            slog::debug << slog::endl;
        }
    }
}

void rawOutputSpoof(const int idx, const cv::Mat& out_landmark) {
    const float as_r = out_landmark.ptr<float>()[0] * 100;
    slog::debug << "[" << idx << "] element, real face probability = " << as_r << slog::endl;
}

float calcMean(const cv::Mat& src) {
    cv::Mat tmp;
    cv::cvtColor(src, tmp, cv::COLOR_BGR2GRAY);
    cv::Scalar mean = cv::mean(tmp);

    return static_cast<float>(mean[0]);
}

void faceDataUpdate(const cv::Mat& frame,
                    Face::Ptr& face,
                    const cv::Rect& face_rect,
                    std::list<Face::Ptr>& prev_faces,
                    const std::vector<cv::Rect>& face_hub,
                    size_t& id,
                    bool no_smooth) {
    // Face update
    cv::Rect rect = face_rect & cv::Rect({0, 0}, frame.size());

    if (!no_smooth) {
        face = matchFace(rect, prev_faces);
        float intensity_mean = calcMean(frame(rect));
        intensity_mean += 1.0;

        if ((face == nullptr) || ((std::abs(intensity_mean - face->_intensity_mean) / face->_intensity_mean) > 0.07f)) {
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

void ageGenderDataUpdate(const Face::Ptr& face, const cv::Mat& out_age, const cv::Mat& out_gender) {
    const float* age_data = out_age.ptr<float>();
    const float* gender_data = out_gender.ptr<float>();

    const float maleProb = gender_data[1];
    const float age = age_data[0] * 100;

    face->updateGender(maleProb);
    face->updateAge(age);
}

void headPoseDataUpdate(const Face::Ptr& face,
                        const cv::Mat& out_y_fc,
                        const cv::Mat& out_p_fc,
                        const cv::Mat& out_r_fc) {
    const float* y_data = out_y_fc.ptr<float>();
    const float* p_data = out_p_fc.ptr<float>();
    const float* r_data = out_r_fc.ptr<float>();

    face->updateHeadPose(y_data[0], p_data[0], r_data[0]);
}

void emotionsDataUpdate(const Face::Ptr& face, const cv::Mat& out_emotion) {
    const float* em_data = out_emotion.ptr<float>();

    std::map<std::string, float> em_val_map;
    for (size_t i = 0; i < EMOTION_VECTOR.size(); i++) {
        em_val_map[EMOTION_VECTOR[i]] = em_data[i];
    }

    face->updateEmotions(em_val_map);
}

void landmarksDataUpdate(const Face::Ptr& face, const cv::Mat& out_landmark) {
    const float* lm_data = out_landmark.ptr<float>();
    const size_t n_lm = 70;
    std::vector<float> normedLandmarks(&lm_data[0], &lm_data[n_lm]);
    face->updateLandmarks(normedLandmarks);
}

void ASpoofDataUpdate(const Face::Ptr& face, const cv::Mat& out_a_spoof) {
    const float* as_data = out_a_spoof.ptr<float>();
    const auto real_face_conf = as_data[0] * 100;
    face->updateRealFaceConfidence(real_face_conf);
}
}  // namespace

int main(int argc, char* argv[]) {
    std::set_terminate(catcher);
    parse(argc, argv);
    PerformanceMetrics metrics;
    /** ---------------- Graph of demo ---------------- **/
    cv::GMat in;

    cv::GMat detections = cv::gapi::infer<Faces>(in);

    cv::GOpaque<cv::Size> sz = cv::gapi::streaming::size(in);
    cv::GArray<cv::Rect> faces_rects = cv::gapi::parseSSD(detections, sz, static_cast<float>(FLAGS_t), false, false);
    cv::GArray<cv::Rect> faces = PostProc::on(faces_rects, sz, FLAGS_bb_enlarge_coef, FLAGS_dx_coef, FLAGS_dy_coef);
    auto outs = GOut(cv::gapi::copy(in), detections, faces);

    cv::GArray<cv::GMat> ages, genders;
    if (!FLAGS_mag.empty()) {
        std::tie(ages, genders) = cv::gapi::infer<AgeGender>(faces, in);
        outs += GOut(ages, genders);
    }

    cv::GArray<cv::GMat> y_fc, p_fc, r_fc;
    if (!FLAGS_mhp.empty()) {
        std::tie(y_fc, p_fc, r_fc) = cv::gapi::infer<HeadPose>(faces, in);
        outs += GOut(y_fc, p_fc, r_fc);
    }

    cv::GArray<cv::GMat> emotions;
    if (!FLAGS_mem.empty()) {
        emotions = cv::gapi::infer<Emotions>(faces, in);
        outs += GOut(emotions);
    }

    cv::GArray<cv::GMat> landmarks;
    if (!FLAGS_mlm.empty()) {
        landmarks = cv::gapi::infer<FacialLandmark>(faces, in);
        outs += GOut(landmarks);
    }

    cv::GArray<cv::GMat> a_spoof;
    if (!FLAGS_mam.empty()) {
        a_spoof = cv::gapi::infer<ASpoof>(faces, in);
        outs += GOut(a_spoof);
    }
    auto pipeline = cv::GComputation(cv::GIn(in), std::move(outs));
    /** ---------------- End of graph ---------------- **/
    /** Configure networks **/
    auto det_net = cv::gapi::ov::Params<Faces>{
        FLAGS_m,  // path to model
        fileNameNoExt(FLAGS_m) + ".bin",  // path to weights
        FLAGS_d  // device to use
    };
    slog::info << "The Face Detection model " << FLAGS_m << " is loaded to " << FLAGS_d << " device." << slog::endl;

    // clang-format off
    auto age_net =
        cv::gapi::ov::Params<AgeGender>{
            FLAGS_mag,  // path to model
            fileNameNoExt(FLAGS_mag) + ".bin",  // path to weights
            FLAGS_dag  // device to use
        }.cfgOutputLayers({"age_conv3", "prob"});
    // clang-format on

    if (!FLAGS_mag.empty()) {
        slog::info << "The Age/Gender Recognition model " << FLAGS_mag << " is loaded to " << FLAGS_dag << " device."
                   << slog::endl;
    } else {
        slog::info << "Age/Gender Recognition DISABLED." << slog::endl;
    }

    // clang-format off
    auto hp_net =
        cv::gapi::ov::Params<HeadPose>{
            FLAGS_mhp,  // path to model
            fileNameNoExt(FLAGS_mhp) + ".bin",  // path to weights
            FLAGS_dhp  // device to use
        }.cfgOutputLayers({"angle_y_fc", "angle_p_fc", "angle_r_fc"});
    // clang-format on

    if (!FLAGS_mhp.empty()) {
        slog::info << "The Head Pose Estimation model " << FLAGS_mhp << " is loaded to " << FLAGS_dhp << " device."
                   << slog::endl;
    } else {
        slog::info << "Head Pose Estimation DISABLED." << slog::endl;
    }

    // clang-format off
    auto lm_net =
        cv::gapi::ov::Params<FacialLandmark>{
            FLAGS_mlm,  // path to model
            fileNameNoExt(FLAGS_mlm) + ".bin",  // path to weights
            FLAGS_dlm  // device to use
        }.cfgOutputLayers({"align_fc3"});
    // clang-format on

    if (!FLAGS_mlm.empty()) {
        slog::info << "The Facial Landmarks Estimation model " << FLAGS_mlm << " is loaded to " << FLAGS_dlm
                   << " device." << slog::endl;
    } else {
        slog::info << "Facial Landmarks Estimation DISABLED." << slog::endl;
    }

    auto am_net = cv::gapi::ov::Params<ASpoof>{
        FLAGS_mam,  // path to model
        fileNameNoExt(FLAGS_mam) + ".bin",  // path to weights
        FLAGS_dam  // device to use
    };
    if (!FLAGS_mam.empty()) {
        slog::info << "The Anti Spoof model " << FLAGS_mam << " is loaded to " << FLAGS_dam << " device." << slog::endl;
    } else {
        slog::info << "Anti Spoof DISABLED." << slog::endl;
    }

    auto emo_net = cv::gapi::ov::Params<Emotions>{
        FLAGS_mem,  // path to model
        fileNameNoExt(FLAGS_mem) + ".bin",  // path to weights
        FLAGS_dem  // device to use
    };
    if (!FLAGS_mem.empty()) {
        slog::info << "The Emotions Recognition model " << FLAGS_mem << " is loaded to " << FLAGS_dem << " device."
                   << slog::endl;
    } else {
        slog::info << "Emotions Recognition DISABLED." << slog::endl;
    }

    /** Custom kernels **/
    auto kernels = cv::gapi::kernels<OCVPostProc>();
    auto networks = cv::gapi::networks(det_net, age_net, hp_net, lm_net, emo_net, am_net);
    auto stream = pipeline.compileStreaming(cv::compile_args(kernels, networks));

    /** Output containers for results **/
    cv::Mat frame, ssd_res;
    std::vector<cv::Rect> face_hub;
    auto out_vector = cv::gout(frame, ssd_res, face_hub);

    std::vector<cv::Mat> out_ages, out_genders;
    if (!FLAGS_mag.empty())
        out_vector += cv::gout(out_ages, out_genders);

    std::vector<cv::Mat> out_y_fc, out_p_fc, out_r_fc;
    if (!FLAGS_mhp.empty())
        out_vector += cv::gout(out_y_fc, out_p_fc, out_r_fc);

    std::vector<cv::Mat> out_emotions;
    if (!FLAGS_mem.empty())
        out_vector += cv::gout(out_emotions);

    std::vector<cv::Mat> out_landmarks;
    if (!FLAGS_mlm.empty())
        out_vector += cv::gout(out_landmarks);

    std::vector<cv::Mat> out_a_spoof;
    if (!FLAGS_mam.empty())
        out_vector += cv::gout(out_a_spoof);

    Visualizer::Ptr visualizer = std::make_shared<Visualizer>(!FLAGS_mag.empty(),
                                                              !FLAGS_mem.empty(),
                                                              !FLAGS_mhp.empty(),
                                                              !FLAGS_mlm.empty(),
                                                              !FLAGS_mam.empty());

    std::list<Face::Ptr> out_faces;
    std::ostringstream out;
    size_t id = 0;

    const cv::Point THROUGHPUT_METRIC_POSITION{10, 30};
    std::unique_ptr<Presenter> presenter;

    /** ---------------- The execution part ---------------- **/
    std::shared_ptr<ImagesCapture> cap = openImagesCapture(FLAGS_i, FLAGS_loop, read_type::safe, 0);
    stream.setSource<custom::CommonCapSrc>(cap);

    /** Save output result **/
    LazyVideoWriter videoWriter{FLAGS_o, cap->fps(), FLAGS_lim};

    bool isStart = true;
    const auto startTime = std::chrono::steady_clock::now();
    stream.start();
    while (stream.pull(cv::GRunArgsP(out_vector))) {
        if (!FLAGS_mem.empty() && FLAGS_show_emotion_bar) {
            visualizer->enableEmotionBar(frame.size(), EMOTION_VECTOR);
        }

        /** Init presenter **/
        if (presenter == nullptr) {
            cv::Size graphSize{static_cast<int>(frame.rows / 4), 60};
            presenter.reset(new Presenter(FLAGS_u, THROUGHPUT_METRIC_POSITION.y + 15, graphSize));
        }

        /**  Postprocessing **/
        std::list<Face::Ptr> prev_faces;

        if (FLAGS_smooth) {
            prev_faces.insert(prev_faces.begin(), out_faces.begin(), out_faces.end());
        }

        out_faces.clear();

        /** Raw output of detected faces **/
        if (FLAGS_r) {
            rawOutputDetections(ssd_res, frame.size(), FLAGS_t);
        }

        /** For every detected face **/
        for (size_t i = 0; i < face_hub.size(); i++) {
            Face::Ptr face;

            cv::Rect rect = face_hub[i] & cv::Rect({0, 0}, frame.size());
            faceDataUpdate(frame, face, rect, prev_faces, face_hub, id, FLAGS_smooth);

            if (!FLAGS_mag.empty()) {
                ageGenderDataUpdate(face, out_ages[i], out_genders[i]);
                if (FLAGS_r)
                    rawOutputAgeGender(i, out_ages[i], out_genders[i]);
            }

            if (!FLAGS_mem.empty()) {
                emotionsDataUpdate(face, out_emotions[i]);
                if (FLAGS_r)
                    rawOutputEmotions(i, out_emotions[i]);
            }

            if (!FLAGS_mhp.empty()) {
                headPoseDataUpdate(face, out_y_fc[i], out_p_fc[i], out_r_fc[i]);
                if (FLAGS_r)
                    rawOutputHeadpose(i, out_y_fc[i], out_p_fc[i], out_r_fc[i]);
            }

            if (!FLAGS_mlm.empty()) {
                landmarksDataUpdate(face, out_landmarks[i]);
                if (FLAGS_r)
                    rawOutputLandmarks(i, out_landmarks[i]);
            }

            if (!FLAGS_mam.empty()) {
                ASpoofDataUpdate(face, out_a_spoof[i]);
                if (FLAGS_r)
                    rawOutputSpoof(i, out_a_spoof[i]);
            }

            /** End of face postprocessing **/
            out_faces.push_back(face);
        }

        /** Drawing faces **/
        visualizer->draw(frame, out_faces);

        presenter->drawGraphs(frame);
        if (isStart) {
            metrics.update(startTime,
                           frame,
                           {10, 22},
                           cv::FONT_HERSHEY_COMPLEX,
                           0.65,
                           {200, 10, 10},
                           2,
                           PerformanceMetrics::MetricTypes::FPS);
            isStart = false;
        } else {
            metrics.update({},
                           frame,
                           {10, 22},
                           cv::FONT_HERSHEY_COMPLEX,
                           0.65,
                           {200, 10, 10},
                           2,
                           PerformanceMetrics::MetricTypes::FPS);
        }

        /** Visualizing results **/
        if (FLAGS_show) {
            cv::imshow(argv[0], frame);

            int key = cv::waitKey(1);
            if ('P' == key || 'p' == key || '0' == key || ' ' == key) {
                key = cv::waitKey(0);
            }
            if (27 == key || 'Q' == key || 'q' == key) {
                stream.stop();
            } else {
                presenter->handleKey(key);
            }
        }

        videoWriter.write(frame);
    }

    slog::info << "Metrics report:" << slog::endl;
    slog::info << "\tFPS: " << std::fixed << std::setprecision(1) << metrics.getTotal().fps << slog::endl;
    slog::info << presenter->reportMeans() << slog::endl;

    return 0;
}
