// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/infer/ov.hpp>
#include <opencv2/gapi/infer/parsers.hpp>
#include <openvino/openvino.hpp>

#include "actions.hpp"
#include "custom_kernels.hpp"
#include "kernel_packages.hpp"
#include "recognizer.hpp"

namespace nets {
G_API_NET(FaceDetector, <cv::GMat(cv::GMat)>, "face-detector");
G_API_NET(LandmarksDetector, <cv::GMat(cv::GMat)>, "landmarks-detector");
G_API_NET(FaceReidentificator, <cv::GMat(cv::GMat)>, "face-reidentificator");
using PAInfo = std::tuple<cv::GMat, cv::GMat, cv::GMat, cv::GMat, cv::GMat, cv::GMat, cv::GMat>;
G_API_NET(PersonDetActionRec, <PAInfo(cv::GMat)>, "person-detection-action-recognition");
}  // namespace nets

namespace config {
inline char separator() {
#ifdef _WIN32
    return '\\';
#else
    return '/';
#endif
}

bool fileExists(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

std::string folderName(const std::string& path) {
    size_t found_pos;
    found_pos = path.find_last_of(separator());
    if (found_pos != std::string::npos)
        return path.substr(0, found_pos);
    return std::string(".") + separator();
}

std::vector<std::string> parseActionLabels(const std::string& in_str) {
    std::vector<std::string> labels;
    std::string label;
    std::istringstream stream_to_split(in_str);
    while (std::getline(stream_to_split, label, ',')) {
        labels.push_back(label);
    }
    return labels;
}

FaceRecognizerConfig getRecConfig(const bool greedy_reid_matching, const double t_reid) {
    FaceRecognizerConfig rec_config;
    rec_config.reid_threshold = t_reid;
    rec_config.greedy_reid_matching = greedy_reid_matching;
    return rec_config;
}

bool isNetForSixActions(const std::string& model_path) {
    CV_Assert(!model_path.empty());
    return model_path.at(model_path.size() - 5) == '6';
}

std::shared_ptr<ActionDetection> createActDetPtr(const std::string& ad_model_path,
                                                 const cv::Size frame_size,
                                                 const size_t actions_map_size,
                                                 const double t_ad,
                                                 const double t_ar) {
    // Load action detector
    ActionDetectorConfig action_config;
    action_config.net_with_six_actions = config::isNetForSixActions(ad_model_path);
    action_config.detection_confidence_threshold = static_cast<float>(t_ad);
    action_config.action_confidence_threshold = static_cast<float>(t_ar);
    action_config.num_action_classes = actions_map_size;
    action_config.input_height = frame_size.height;
    action_config.input_width = frame_size.width;
    return std::make_shared<ActionDetection>(action_config);
}

detection::DetectorConfig getDetConfig(const double exp_r_fd) {
    // Load face detector
    detection::DetectorConfig face_config;
    face_config.increase_scale_x = static_cast<float>(exp_r_fd);
    face_config.increase_scale_y = static_cast<float>(exp_r_fd);
    return face_config;
}

struct ArgsFlagsPack {
    std::string teacher_id;
    double min_ad;
    double d_ad;
    std::string student_ac;
    std::string top_ac;
    std::string teacher_ac;
    std::string top_id;
    int a_top;
    int ss_t;
    bool no_show;
};

struct NetsFlagsPack {
    std::string m_act;
    std::string m_fd;
    std::string m_lm;
    std::string m_reid;
    std::string d_act;
    std::string d_fd;
    std::string d_lm;
    std::string d_reid;
    int inh_fd;
    int inw_fd;
};

std::tuple<ConstantParams, TrackerParams, TrackerParams> getGraphArgs(const std::string& video_path,
                                                                      const cv::Size& frame_size,
                                                                      const double fps,
                                                                      const ArgsFlagsPack& flags) {
    ConstantParams const_params;
    const_params.teacher_id = flags.teacher_id;
    const_params.actions_type = flags.teacher_id.empty() ? flags.a_top > 0 ? TOP_K : STUDENT : TEACHER;
    const_params.actions_map = const_params.actions_type == STUDENT
                                   ? parseActionLabels(flags.student_ac)
                                   : const_params.actions_type == TOP_K ? parseActionLabels(flags.top_ac)
                                                                        : parseActionLabels(flags.teacher_ac);
    const_params.top_action_id = static_cast<int>(
        const_params.actions_type == TOP_K
            ? std::distance(const_params.actions_map.begin(),
                            find(const_params.actions_map.begin(), const_params.actions_map.end(), flags.top_id))
            : -1);
    if (const_params.actions_type == TOP_K &&
        (const_params.top_action_id < 0 ||
         const_params.top_action_id >= static_cast<int>(const_params.actions_map.size()))) {
        slog::err << "Cannot find target action: " << flags.top_id << slog::endl;
    }
    const auto num_top_persons = const_params.actions_type == TOP_K ? flags.a_top : -1;
    const_params.draw_ptr.reset(new DrawingHelper(flags.no_show, num_top_persons));
    const_params.video_path = video_path;
    const_params.smooth_window_size = static_cast<int>(fps * flags.d_ad);
    const_params.smooth_min_length = static_cast<int>(fps * flags.min_ad);
    const_params.top_flag = flags.a_top;
    const_params.draw_ptr->GetNewFrameSize(frame_size);

    /** Create tracker parameters for reidentification **/
    TrackerParams tracker_reid_params;
    tracker_reid_params.min_track_duration = 1;
    tracker_reid_params.forget_delay = 150;
    tracker_reid_params.affinity_thr = 0.8f;
    tracker_reid_params.averaging_window_size_for_rects = 1;
    tracker_reid_params.averaging_window_size_for_labels = std::numeric_limits<int>::max();
    tracker_reid_params.bbox_heights_range = cv::Vec2f(10, 1080);
    tracker_reid_params.drop_forgotten_tracks = false;
    tracker_reid_params.max_num_objects_in_track = std::numeric_limits<int>::max();
    tracker_reid_params.objects_type = "face";

    /** Create tracker parameters for action recognition **/
    TrackerParams tracker_action_params;
    tracker_action_params.min_track_duration = 8;
    tracker_action_params.forget_delay = 150;
    tracker_action_params.affinity_thr = 0.9f;
    tracker_action_params.averaging_window_size_for_rects = 5;
    tracker_action_params.averaging_window_size_for_labels =
        flags.ss_t > 0 ? flags.ss_t : const_params.actions_type == TOP_K ? 5 : 1;
    tracker_action_params.bbox_heights_range = cv::Vec2f(10, 2160);
    tracker_action_params.drop_forgotten_tracks = false;
    tracker_action_params.max_num_objects_in_track = std::numeric_limits<int>::max();
    tracker_action_params.objects_type = "action";

    return std::make_tuple(const_params, tracker_reid_params, tracker_action_params);
}

void printInfo(const NetsFlagsPack& flags, std::string& teacher_id, std::string& top_id) {
    slog::info << ov::get_openvino_version() << slog::endl;
    if (!teacher_id.empty() && !top_id.empty()) {
        slog::err << "Cannot run simultaneously teacher action and top-k students recognition." << slog::endl;
    }
}

void configNets(const NetsFlagsPack& flags, cv::gapi::GNetPackage& networks, cv::Size& act_net_in_size, cv::Scalar& reid_net_in_size) {
    if (!flags.m_act.empty()) {
        const std::vector<std::string> action_detector_5 = {"mbox_loc1/out/conv/flat",
                                                            "mbox_main_conf/out/conv/flat/softmax/flat",
                                                            "mbox/priorbox",
                                                            "out/anchor1",
                                                            "out/anchor2",
                                                            "out/anchor3",
                                                            "out/anchor4"};
        const std::vector<std::string> action_detector_6 = {"ActionNet/out_detection_loc",
                                                            "ActionNet/out_detection_conf",
                                                            "ActionNet/action_heads/out_head_1_anchor_1",
                                                            "ActionNet/action_heads/out_head_2_anchor_1",
                                                            "ActionNet/action_heads/out_head_2_anchor_2",
                                                            "ActionNet/action_heads/out_head_2_anchor_3",
                                                            "ActionNet/action_heads/out_head_2_anchor_4"};
        /** Create action detector net's parameters **/
        std::vector<std::string> outputBlobList =
            isNetForSixActions(flags.m_act) ? action_detector_6 : action_detector_5;
        // clang-format off
        auto action_net =
            cv::gapi::ov::Params<nets::PersonDetActionRec>{
                flags.m_act,
                fileNameNoExt(flags.m_act) + ".bin",
                flags.d_act,
            }.cfgOutputLayers(outputBlobList);
        // clang-format on

        networks += cv::gapi::networks(action_net);
        ov::Core core;
        const ov::Output<ov::Node>& input = core.read_model(flags.m_act)->input();
        const ov::Shape& in_shape = input.get_shape();
        act_net_in_size = {int(in_shape[2]), int(in_shape[1])};
        slog::info << "The Person/Action Detection model " << flags.m_act << " is loaded to " << flags.d_act
                   << " device." << slog::endl;
    } else {
        slog::info << "Person/Action Detection DISABLED." << slog::endl;
    }
    if (!flags.m_fd.empty()) {
        /** Create face detector net's parameters **/
        // clang-format off
        auto det_net =
            cv::gapi::ov::Params<nets::FaceDetector>{
                flags.m_fd,
                fileNameNoExt(flags.m_fd) + ".bin",
                flags.d_fd,
            }.cfgReshape({1u, 3u, static_cast<size_t>(flags.inh_fd), static_cast<size_t>(flags.inw_fd)});
        // clang-format on

        networks += cv::gapi::networks(det_net);
        slog::info << "The Face Detection model" << flags.m_fd << " is loaded to " << flags.d_fd << " device."
                   << slog::endl;
    } else {
        slog::info << "Face Detection DISABLED." << slog::endl;
    }

    if (!flags.m_fd.empty() && !flags.m_reid.empty() && !flags.m_lm.empty()) {
        /** Create landmarks detector net's parameters **/
        auto landm_net = cv::gapi::ov::Params<nets::LandmarksDetector>{
            flags.m_lm,
            fileNameNoExt(flags.m_lm) + ".bin",
            flags.d_lm,
        };
        if (!flags.m_lm.empty()) {
            slog::info << "The Facial Landmarks Regression model" << flags.m_lm << " is loaded to " << flags.d_lm
                       << " device." << slog::endl;
        } else {
            slog::info << "Facial Landmarks Regression DISABLED." << slog::endl;
        }
        /** Create reidentification net's parameters **/
        auto reident_net = cv::gapi::ov::Params<nets::FaceReidentificator>{
            flags.m_reid,
            fileNameNoExt(flags.m_reid) + ".bin",
            flags.d_reid,
        };
        networks += cv::gapi::networks(landm_net, reident_net);
        if (!flags.m_reid.empty()) {
            slog::info << "The Face Re-Identification model " << flags.m_reid << " is loaded to " << flags.d_reid
                       << " device." << slog::endl;
        } else {
            slog::info << "Face Re-Identification DISABLED." << slog::endl;
        }
        ov::Core core;
        const ov::Output<ov::Node>& input = core.read_model(flags.m_reid)->input();
        const ov::Shape& in_shape = input.get_shape();
        reid_net_in_size = {static_cast<double>(in_shape[0]),
                            static_cast<double>(in_shape[1]),
                            static_cast<double>(in_shape[2]),
                            static_cast<double>(in_shape[3])};
    }
}
}  // namespace config

namespace preparation {
struct FGFlagsPack {
    bool greedy_reid_matching;
    double t_reid;
    std::string fg;
    bool crop_gallery;
    double t_reg_fd;
    int min_size_fr;
    double exp_r_fd;
};

std::shared_ptr<FaceRecognizer> processingFaceGallery(const cv::gapi::GNetPackage& gallery_networks,
                                                      const FGFlagsPack& flags,
                                                      const cv::Scalar& reid_net_in_size) {
    // Face gallery processing
    std::vector<int> idx_to_id;
    std::vector<GalleryObject> identities;
    const auto ids_list = flags.fg;
    std::shared_ptr<detection::FaceDetection> face_det_ptr =
        std::make_shared<detection::FaceDetection>(config::getDetConfig(flags.exp_r_fd));
    if (!ids_list.empty()) {
        /** ---------------- Gallery graph of demo ---------------- **/
        /** Input is one face from gallery **/
        cv::GMat in;
        cv::GArray<cv::Rect> rect;

        /** Crop face from image **/
        if (flags.crop_gallery) {
            /** Detect face **/
            cv::GMat detections = cv::gapi::infer<nets::FaceDetector>(in);
            cv::GOpaque<cv::Size> sz = cv::gapi::streaming::size(in);
            cv::GArray<cv::Rect> face_rect =
                cv::gapi::parseSSD(detections, sz, static_cast<float>(flags.t_reg_fd), true, true);
            rect = custom::FaceDetectorPostProc::on(in, face_rect, face_det_ptr);
        } else {
            /** Else ROI is equal to image size **/
            rect = custom::GetRectFromImage::on(in);
        }
        /** Get landmarks by ROI **/
        cv::GArray<cv::GMat> landmarks = cv::gapi::infer<nets::LandmarksDetector>(rect, in);

        /** Align face by landmarks **/
        cv::GScalar net_size(reid_net_in_size);
        cv::GArray<cv::GMat> align_faces = custom::AlignFacesForReidentification::on(in, landmarks, rect, net_size);

        /** Get face identities metrics **/
        cv::GArray<cv::GMat> embeddings = cv::gapi::infer2<nets::FaceReidentificator>(in, align_faces);

        /** Pipeline's input and outputs**/
        cv::GComputation gallery_pp(cv::GIn(in), cv::GOut(rect, embeddings));
        /** ---------------- End of gallery graph ---------------- **/
        cv::FileStorage fs(ids_list, cv::FileStorage::Mode::READ);
        cv::FileNode fn = fs.root();
        int id = 0;
        for (const auto& item : fn) {
            std::string label = item.name();
            std::vector<cv::Mat> out_embeddings;
            // Please, note that the case when there are more than one image in gallery
            // for a person might not work properly with the current implementation
            // of the demo.
            // Remove this assert by your own risk.
            CV_Assert(item.size() == 1);

            for (const auto& item_e : item) {
                cv::Mat image;
                std::vector<cv::Mat> emb;
                if (config::fileExists(item_e.string())) {
                    image = cv::imread(item_e.string());
                } else {
                    image = cv::imread(config::folderName(ids_list) + config::separator() + item_e.string());
                }
                CV_Assert(!image.empty());
                std::vector<cv::Rect> out_rect;
                gallery_pp.apply(cv::gin(image),
                                 cv::gout(out_rect, emb),
                                 cv::compile_args(custom::kernels(), gallery_networks));
                CV_Assert(emb.size() == 1);
                CV_Assert(out_rect.size() == 1);
                // NOTE: RegistrationStatus analog check
                if (!out_rect.empty() && (out_rect[0].width > flags.min_size_fr) &&
                    (out_rect[0].height > flags.min_size_fr)) {
                    out_embeddings.emplace_back(emb.front().reshape(1, {static_cast<int>(emb.front().total()), 1}));
                    idx_to_id.emplace_back(id);
                    identities.emplace_back(out_embeddings, label, id);
                    ++id;
                }
            }
        }
        slog::debug << "Face reid gallery size: " << identities.size() << slog::endl;
    } else {
        slog::warn << "Face reid gallery is empty!" << slog::endl;
    }
    FaceRecognizerConfig rec_config;
    rec_config.reid_threshold = flags.t_reid;
    rec_config.greedy_reid_matching = flags.greedy_reid_matching;
    rec_config.identities = identities;
    rec_config.idx_to_id = idx_to_id;
    return std::make_shared<FaceRecognizer>(rec_config);
}
}  // namespace preparation
