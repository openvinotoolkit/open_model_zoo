// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <stddef.h>

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/gapi/garray.hpp>
#include <opencv2/gapi/gkernel.hpp>
#include <opencv2/gapi/gmat.hpp>
#include <opencv2/gapi/gopaque.hpp>
#include <opencv2/gapi/render/render_types.hpp>

#include "tracker.hpp"

class ActionDetection;
class DrawingHelper;
class FaceRecognizer;
namespace cv {
class GScalar;
namespace detail {
template <typename T>
struct CompileArgTag;
}  // namespace detail
struct GScalarDesc;
}  // namespace cv
namespace detection {
class FaceDetection;
}  // namespace detection
struct DetectedAction;
struct DrawingElements;

/** Parameters of trackers for stateful kernels **/
struct TrackerParamsPack {
    TrackerParams tracker_reid_params;
    TrackerParams tracker_action_params;
};

/** Shared between kernels constants **/
struct ConstantParams {
    std::shared_ptr<DrawingHelper> draw_ptr;
    std::vector<std::string> actions_map;
    std::string teacher_id;
    std::string video_path;
    double actions_type = 0;
    int top_flag = -1;
    size_t work_num_frames = 0;
    size_t total_num_frames = 0;
    int top_action_id = -1;
    size_t num_frames = 0;
    int smooth_window_size = -1;
    int smooth_min_length = -1;
};

/** Face tracking results **/
struct FaceTrack {
    std::vector<TrackedObject> tracked_faces;
    std::vector<std::string> face_labels;
    std::vector<Track> face_tracks;
};

namespace cv {
namespace detail {
template <>
struct CompileArgTag<TrackerParamsPack> {
    static const char* tag() {
        return "custom.get_recognition_result_state_params";
    }
};

template <>
struct CompileArgTag<bool> {
    static const char* tag() {
        return "custom.logger_state_params";
    }
};
}  // namespace detail
}  // namespace cv

using GPrims = cv::GArray<cv::gapi::wip::draw::Prim>;
template <typename T>
using four = std::tuple<T, T, T, T>;

namespace custom {
// clang-format off
    G_API_OP(BoxesAndLabels,
             <GPrims(const cv::GMat,
                     const cv::GOpaque<DrawingElements>,
                     const ConstantParams &)>,
            "custom.boxes_and_labels") {
        static cv::GArrayDesc outMeta(const cv::GMatDesc&,
                                      const cv::GOpaqueDesc&,
                                      const ConstantParams&) {
            return cv::empty_array_desc();
        }
    };
    G_API_OP(RecognizeResultPostProc,
             <std::tuple<cv::GOpaque<DrawingElements>,
                         cv::GOpaque<std::string>,
                         cv::GOpaque<std::string>,
                         cv::GOpaque<std::string>>(cv::GMat,
                                                   cv::GArray<TrackedObject>,
                                                   cv::GOpaque<FaceTrack>,
                                                   cv::GArray<std::string>,
                                                   cv::GOpaque<size_t>,
                                                   ConstantParams)>,
             "custom.processing_result_of_recognize") {
        static four<cv::GOpaqueDesc> outMeta(const cv::GMatDesc& in,
                                             const cv::GArrayDesc&,
                                             const cv::GOpaqueDesc&,
                                             const cv::GArrayDesc&,
                                             const cv::GOpaqueDesc&,
                                             const ConstantParams&) {
            return std::make_tuple(cv::empty_gopaque_desc(),
                                   cv::empty_gopaque_desc(),
                                   cv::empty_gopaque_desc(),
                                   cv::empty_gopaque_desc());
        }
    };

    G_API_OP(TopAction,
             <std::tuple<cv::GOpaque<DrawingElements>, cv::GMat>(cv::GMat,
                                                                 cv::GArray<TrackedObject>,
                                                                 ConstantParams)>,
            "sample.custom.rising_hand_processing") {
        static std::tuple<cv::GOpaqueDesc, cv::GMatDesc> outMeta(const cv::GMatDesc& in,
                                                                 const cv::GArrayDesc&,
                                                                 const ConstantParams&) {
            return std::make_tuple(cv::empty_gopaque_desc(), in);
        }
    };

    G_API_OP(FaceDetectorPostProc,
             <cv::GArray<cv::Rect>(cv::GMat,
                                   cv::GArray<cv::Rect>,
                                   std::shared_ptr<detection::FaceDetection>)>,
             "custom.fd_postproc") {
        static cv::GArrayDesc outMeta(const cv::GMatDesc&,
                                      const cv::GArrayDesc&,
                                      const std::shared_ptr<detection::FaceDetection>&) {
            return cv::empty_array_desc();
        }
    };

    G_API_OP(GetRectFromImage,
             <cv::GArray<cv::Rect>(cv::GMat)>,
             "custom.get_rect_from_image") {
        static cv::GArrayDesc outMeta(const cv::GMatDesc&) {
            return cv::empty_array_desc();
        }
    };

    G_API_OP(PersonDetActionRecPostProc,
             <cv::GArray<DetectedAction>(cv::GMat, cv::GMat,
                                         cv::GMat, cv::GMat,
                                         cv::GMat, cv::GMat,
                                         cv::GMat, cv::GMat,
                                         std::shared_ptr<ActionDetection>)>,
             "custom.person_detection_action_recognition_postproc") {
        static cv::GArrayDesc outMeta(const cv::GMatDesc&, const cv::GMatDesc&,
                                      const cv::GMatDesc&, const cv::GMatDesc&,
                                      const cv::GMatDesc&, const cv::GMatDesc&,
                                      const cv::GMatDesc&, const cv::GMatDesc&,
                                      const std::shared_ptr<ActionDetection>&) {
            return cv::empty_array_desc();
        }
    };

    G_API_OP(AlignFacesForReidentification,
             <cv::GArray<cv::GMat>(cv::GMat, cv::GArray<cv::GMat>, cv::GArray<cv::Rect>, cv::GScalar)>,
             "custom.align_faces_for_reidentification") {
        static cv::GArrayDesc outMeta(const cv::GMatDesc&,
                                      const cv::GArrayDesc&,
                                      const cv::GArrayDesc&,
                                      const cv::GScalarDesc&) {
            return cv::empty_array_desc();
        }
    };

    G_API_OP(GetRecognitionResult,
             <std::tuple<cv::GArray<TrackedObject>,
                         cv::GOpaque<FaceTrack>,
                         cv::GOpaque<size_t>>(cv::GMat,
                                              cv::GArray<cv::Rect>,
                                              cv::GArray<DetectedAction>,
                                              cv::GArray<cv::GMat>,
                                              std::shared_ptr<FaceRecognizer>,
                                              ConstantParams)>,
             "custom.get_recognition_result") {
        static std::tuple<cv::GArrayDesc,
                          cv::GOpaqueDesc,
                          cv::GOpaqueDesc> outMeta(const cv::GMatDesc&,
                                                   const cv::GArrayDesc&,
                                                   const cv::GArrayDesc&,
                                                   const cv::GArrayDesc&,
                                                   const std::shared_ptr<FaceRecognizer>&,
                                                   const ConstantParams&) {
            return std::make_tuple(cv::empty_array_desc(),
                                   cv::empty_gopaque_desc(),
                                   cv::empty_gopaque_desc());
        }
    };

    G_API_OP(GetActionTopHandsDetectionResult,
             <cv::GArray<TrackedObject>(cv::GMat,
                                        cv::GArray<DetectedAction>)>,
             "custom.get_action_detection_result_for_top_k_first_hands") {
        static cv::GArrayDesc outMeta(const cv::GMatDesc&, const cv::GArrayDesc&) {
            return cv::empty_array_desc();
        }
    };
// clang-format on
}  // namespace custom
