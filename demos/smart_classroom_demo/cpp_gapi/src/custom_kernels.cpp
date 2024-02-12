// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom_kernels.hpp"

#include <algorithm>
#include <map>

#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/gcommon.hpp>
#include <opencv2/gapi/util/optional.hpp>
#include <opencv2/imgproc.hpp>

#include <utils/slog.hpp>

#include "action_detector.hpp"
#include "actions.hpp"
#include "detector.hpp"
#include "drawing_helper.hpp"
#include "face_reid.hpp"
#include "kernel_packages.hpp"
#include "logger.hpp"
#include "recognizer.hpp"

/** State parameters for RecognizeResultPostProc stateful kernel **/
struct PostProcState {
    PostProcState(const bool write) : logger(write), update_logs(write) {}
    DetectionsLogger logger;
    int teacher_track_id = -1;
    std::vector<std::map<int, int>> face_obj_id_to_action_maps;
    bool update_logs = false;
};

/** State parameter for TopAction stateful kernel **/
struct TopKState {
    std::map<int, int> top_k_obj_ids;
};

/** State parameters for GetRecognitionResult stateful kernel **/
struct TrackerState {
    TrackerState(TrackerParams tracker_reid_params, TrackerParams tracker_action_params)
        : tracker_reid(Tracker(tracker_reid_params)),
          tracker_action(Tracker(tracker_action_params)) {}
    Tracker tracker_reid;
    Tracker tracker_action;
};

/** Colors for labels and ROI **/
const cv::Scalar red_color = CV_RGB(255, 0, 0);
const cv::Scalar white_color = CV_RGB(255, 255, 255);

// clang-format off
GAPI_OCV_KERNEL(OCVFaceDetectorPostProc, custom::FaceDetectorPostProc) {
    static void run(const cv::Mat& in,
                    const std::vector<cv::Rect>& rois,
                    const std::shared_ptr<detection::FaceDetection> &face_det,
                    std::vector<cv::Rect>& out_rects) {
        face_det->truncateRois(in, rois, out_rects);
    }
};

GAPI_OCV_KERNEL(OCVGetRectFromImage, custom::GetRectFromImage) {
    static void run(const cv::Mat& in_image,
                    std::vector<cv::Rect>& out_rects) {
        out_rects.emplace_back(cv::Rect(0, 0, in_image.cols, in_image.rows));
    }
};

GAPI_OCV_KERNEL(OCVAlignFacesForReidentification,
                custom::AlignFacesForReidentification) {
    static void run(const cv::Mat& in,
                    const std::vector<cv::Mat>& landmarks,
                    const std::vector<cv::Rect>& face_rois,
                    const cv::Scalar_<int>& net_input_size,
                          std::vector<cv::Mat>& out_images) {
        cv::Mat out_image = in.clone();
        out_images.clear();
        for (const auto& rect : face_rois) {
            out_images.emplace_back(out_image(rect));
        }
        AlignFaces(&out_images, &const_cast<std::vector<cv::Mat>&>(landmarks));
        /** Preprocessing for CNN input **/
        for (auto& image : out_images) {
            cv::Mat rsz, cvt;
            cv::resize(image, rsz, cv::Size(static_cast<int>(net_input_size[2]),
                static_cast<int>(net_input_size[3])));  // resize
            rsz.convertTo(cvt, CV_32F);                // to F32 type
            image.create(cv::Size(cvt.cols, cvt.rows * cvt.channels()), CV_32F);
            std::vector<cv::Mat> planes;
            for (int i = 0; i < cvt.channels(); ++i) {
                planes.push_back(image.rowRange(i * cvt.rows, (i + 1) * cvt.rows));
            }
            cv::split(cvt, planes);                     // to NCHW
            std::vector<int> new_dims =
                { static_cast<int>(net_input_size[0]),
                static_cast<int>(net_input_size[1]),
                static_cast<int>(net_input_size[2]),
                static_cast<int>(net_input_size[3]) };
            image = image.reshape(1, new_dims);  // reshape to CNN input
        }
    }
};

GAPI_OCV_KERNEL(OCVPersonDetActionRecPostProc, custom::PersonDetActionRecPostProc) {
    static void run(const cv::Mat& in_frame,
                    const cv::Mat& in_ssd_local,
                    const cv::Mat& in_ssd_conf,
                    const cv::Mat& in_ssd_priorbox,
                    const cv::Mat& in_ssd_anchor1,
                    const cv::Mat& in_ssd_anchor2,
                    const cv::Mat& in_ssd_anchor3,
                    const cv::Mat& in_ssd_anchor4,
                    const std::shared_ptr<ActionDetection>& action_det,
                    DetectedActions& out_detections) {
        out_detections = action_det->fetchResults({in_ssd_local,
                                                   in_ssd_conf,
                                                   in_ssd_priorbox,
                                                   in_ssd_anchor1,
                                                   in_ssd_anchor2,
                                                   in_ssd_anchor3,
                                                   in_ssd_anchor4},
                                                  in_frame);
    }
};

GAPI_OCV_KERNEL_ST(OCVGetActionTopHandsDetectionResult,
                   custom::GetActionTopHandsDetectionResult,
                   Tracker) {
    static void setup(const cv::GMatDesc&,
                      const cv::GArrayDesc&,
                      std::shared_ptr<Tracker>& tracker_action,
                      const cv::GCompileArgs& compileArgs) {
        auto trParamsPack = cv::gapi::getCompileArg<TrackerParamsPack>(compileArgs)
            .value_or(TrackerParamsPack{});
        tracker_action = std::make_shared<Tracker>(trParamsPack.tracker_action_params);
    }
    static void run(const cv::Mat& frame,
                    const DetectedActions& actions,
                    TrackedObjects& tracked_actions,
                    Tracker& tracker_action) {
        TrackedObjects tracked_action_objects;
        for (const auto& action : actions) {
            tracked_action_objects.emplace_back(action.rect, action.detection_conf, action.label);
        }
        tracker_action.Process(frame, tracked_action_objects);
        tracked_actions = tracker_action.TrackedDetectionsWithLabels();
    }
};

GAPI_OCV_KERNEL_ST(OCVGetRecognitionResult, custom::GetRecognitionResult, TrackerState) {
    static void setup(const cv::GMatDesc&,
                      const cv::GArrayDesc&,
                      const cv::GArrayDesc&,
                      const cv::GArrayDesc&,
                      const std::shared_ptr<FaceRecognizer>&,
                      const ConstantParams&,
                      std::shared_ptr<TrackerState>& trackers,
                      const cv::GCompileArgs& compileArgs) {
        auto trParamsPack = cv::gapi::getCompileArg<TrackerParamsPack>(compileArgs)
            .value_or(TrackerParamsPack{});
        trackers = std::make_shared<TrackerState>(trParamsPack.tracker_reid_params,
            trParamsPack.tracker_action_params);
    }
    static void run(const cv::Mat& frame,
                    const std::vector<cv::Rect>& faces,
                    const DetectedActions& actions,
                    const std::vector<cv::Mat>& embeddings,
                    const std::shared_ptr<FaceRecognizer>& face_rec,
                    const ConstantParams& params,
                    TrackedObjects& tracked_actions,
                    FaceTrack& face_track,
                    size_t& num_frames,
                    TrackerState &trackers) {
        TrackedObjects tracked_face_objects, tracked_action_objects, tracked_faces;
        std::vector<Track> face_tracks;
        std::vector<std::string> face_labels;
        std::vector<int> ids = face_rec->Recognize(faces, const_cast<std::vector<cv::Mat>&>(embeddings));
        for (size_t i = 0; i < faces.size(); ++i) {
            tracked_face_objects.emplace_back(faces[i], 1.f, ids[i]);
        }
        trackers.tracker_reid.Process(frame, tracked_face_objects);
        tracked_faces = trackers.tracker_reid.TrackedDetectionsWithLabels();

        for (const auto& face : tracked_faces) {
            face_labels.push_back(face_rec->GetLabelByID(face.label));
        }

        for (const auto& action : actions) {
            tracked_action_objects.emplace_back(action.rect, action.detection_conf, action.label);
        }

        trackers.tracker_action.Process(frame, tracked_action_objects);
        tracked_actions = trackers.tracker_action.TrackedDetectionsWithLabels();

        if (!params.actions_type) {
            face_tracks = trackers.tracker_reid.vector_tracks();
        }
        num_frames = trackers.tracker_action.pipeline_idx;
        face_track = {tracked_faces, face_labels, face_tracks};
    }
};

GAPI_OCV_KERNEL_ST(OCVRecognizeResultPostProc, custom::RecognizeResultPostProc, PostProcState) {
    static void setup(const cv::GMatDesc&,
                      const cv::GArrayDesc&,
                      const cv::GOpaqueDesc&,
                      const cv::GArrayDesc&,
                      const cv::GOpaqueDesc&,
                      const ConstantParams&,
                      std::shared_ptr<PostProcState>& post_proc,
                      const cv::GCompileArgs& compileArgs) {
        auto logger_params = cv::gapi::getCompileArg<bool>(compileArgs)
            .value_or(false);
        post_proc = std::make_shared<PostProcState>(logger_params);
    }
    static void run(const cv::Mat& frame,
                    const TrackedObjects& tracked_actions,
                    const FaceTrack& face_track,
                    const std::vector<std::string>& face_id_to_label_map,
                    const size_t& work_num_frames,
                    const ConstantParams& params,
                    DrawingElements& drawing_elements,
                    std::string& stream_log,
                    std::string& stat_log,
                    std::string& det_log,
                    PostProcState& post_proc) {
        int teacher_track_id = -1;
        const int default_action_index = -1;
        std::map<int, int> frame_face_obj_id_to_action;
        size_t labels_step = 0;
        std::vector<cv::Rect> out_rects_det, out_rects_face;
        std::vector<std::string> out_labels_det, out_labels_face;
        post_proc.logger.CreateNextFrameRecord(params.video_path, work_num_frames, frame.cols, frame.rows);
        for (const auto& face : face_track.tracked_faces) {
            std::string face_label = face_track.face_labels.at(labels_step++);
            std::string label_to_draw;
            if (face.label != EmbeddingsGallery::unknown_id)
                label_to_draw += face_label;
            int person_ind = params.draw_ptr->GetIndexOfTheNearestPerson(face, tracked_actions);
            int action_ind = default_action_index;
            if (person_ind >= 0) {
                action_ind = tracked_actions[person_ind].label;
            }

            if (params.actions_type == 0) {
                if (action_ind != default_action_index) {
                    label_to_draw += "[" + params.draw_ptr->GetActionTextLabel(action_ind, params.actions_map) + "]";
                    frame_face_obj_id_to_action[face.object_id] = action_ind;
                    post_proc.logger.AddFaceToFrame(face.rect, face_label, "");
                }
                out_rects_face.emplace_back(face.rect);
                out_labels_face.emplace_back(label_to_draw);
            }
            if ((params.actions_type == 1) && (person_ind >= 0)) {
                teacher_track_id = post_proc.teacher_track_id;
                if (face_label == params.teacher_id) {
                    teacher_track_id = tracked_actions[person_ind].object_id;
                } else if (teacher_track_id == tracked_actions[person_ind].object_id) {
                    teacher_track_id = -1;
                }
            }
        }
        if (params.actions_type == 0) {
            for (const auto& action : tracked_actions) {
                const auto& action_label = params.draw_ptr->GetActionTextLabel(action.label, params.actions_map);
                const auto& text_label = face_track.tracked_faces.empty() ? action_label : "";
                out_labels_det.emplace_back(text_label);
                out_rects_det.emplace_back(action.rect);
                post_proc.logger.AddPersonToFrame(action.rect, action_label, "");
                post_proc.logger.AddDetectionToFrame(action, work_num_frames);
            }
            post_proc.face_obj_id_to_action_maps.push_back(frame_face_obj_id_to_action);
        } else if (teacher_track_id >= 0) {
            auto res_find = std::find_if(tracked_actions.begin(), tracked_actions.end(),
                [teacher_track_id](const TrackedObject& o) { return o.object_id == teacher_track_id; });
            if (res_find != tracked_actions.end()) {
                const auto& tracker_action = *res_find;
                const auto& action_label =
                    params.draw_ptr->GetActionTextLabel(tracker_action.label, params.actions_map);
                out_labels_det.emplace_back(action_label);
                out_rects_det.emplace_back(tracker_action.rect);
                post_proc.logger.AddPersonToFrame(tracker_action.rect, action_label, params.teacher_id);
            }
            post_proc.teacher_track_id = teacher_track_id;
        }
        post_proc.logger.FinalizeFrameRecord();

        if (params.actions_type == 0 && post_proc.update_logs) {
            std::vector<Track> new_face_tracks = UpdateTrackLabelsToBestAndFilterOutUnknowns(face_track.face_tracks);
            std::map<int, int> face_track_id_to_label = post_proc.logger.GetMapFaceTrackIdToLabel(new_face_tracks);
            if (!face_id_to_label_map.empty()) {
                std::map<int, FrameEventsTrack> face_obj_id_to_actions_track;
                post_proc.logger.ConvertActionMapsToFrameEventTracks(post_proc.face_obj_id_to_action_maps,
                                                                     default_action_index,
                                                                     &face_obj_id_to_actions_track);
                const int start_frame = 0;
                const int end_frame = post_proc.face_obj_id_to_action_maps.size();
                std::map<int, RangeEventsTrack> face_obj_id_to_events;
                post_proc.logger.SmoothTracks(face_obj_id_to_actions_track, start_frame, end_frame,
                                              params.smooth_window_size, params.smooth_min_length,
                                              default_action_index, &face_obj_id_to_events);
                slog::debug << " Final ID->events mapping" << slog::endl;
                post_proc.logger.DumpTracks(face_obj_id_to_events,
                                            params.actions_map, face_track_id_to_label,
                                            face_id_to_label_map);
                std::vector<std::map<int, int>> face_obj_id_to_smoothed_action_maps;
                post_proc.logger.ConvertRangeEventsTracksToActionMaps(end_frame, face_obj_id_to_events,
                                                                      &face_obj_id_to_smoothed_action_maps);
                slog::debug << " Final per-frame ID->action mapping" << slog::endl;
                post_proc.logger.DumpDetections(params.video_path, frame.size(), work_num_frames,
                                                new_face_tracks,
                                                face_track_id_to_label,
                                                params.actions_map, face_id_to_label_map,
                                                face_obj_id_to_smoothed_action_maps);
            }
        }
        drawing_elements = {out_rects_det, out_rects_face, out_labels_det, out_labels_face};
        std::tie(stream_log, stat_log, det_log) = post_proc.logger.GetLogResult();
    }
};

GAPI_OCV_KERNEL(OCVBoxesAndLabels, custom::BoxesAndLabels) {
    static void run(const cv::Mat& in,
                    const DrawingElements& drawing_elements,
                    const ConstantParams& params,
                          std::vector<cv::gapi::wip::draw::Prim>& out_prims) {
        out_prims.clear();
        const auto rct = [&params](const cv::Rect &rc, const cv::Scalar &clr) {
            cv::Rect rect_to_draw = rc;
            if (params.draw_ptr->rect_scale_x_ != 1 || params.draw_ptr->rect_scale_y_ != 1) {
                rect_to_draw.x = cvRound(rect_to_draw.x * params.draw_ptr->rect_scale_x_);
                rect_to_draw.y = cvRound(rect_to_draw.y * params.draw_ptr->rect_scale_y_);
                rect_to_draw.height = cvRound(rect_to_draw.height * params.draw_ptr->rect_scale_y_);
                rect_to_draw.width = cvRound(rect_to_draw.width * params.draw_ptr->rect_scale_x_);
            }
            return cv::gapi::wip::draw::Rect(rect_to_draw, clr, 1);
        };
        const auto txt = [](const std::string &str, const cv::Rect &rc, const cv::Scalar &clr) {
            return cv::gapi::wip::draw::Text(str, cv::Point(rc.x, rc.y), cv::FONT_HERSHEY_PLAIN, 1,
                        clr, 1, cv::LINE_AA);
        };
        const auto pad = [](const std::string &str, const cv::Rect &rc, const cv::Scalar &clr) {
            int baseLine = 0;
            const cv::Size label_size =
                cv::getTextSize(str, cv::FONT_HERSHEY_PLAIN, 1, 1, &baseLine);
            return cv::gapi::wip::draw::Rect(cv::Rect(cv::Point(rc.x, rc.y - label_size.height),
                                             cv::Point(rc.x + label_size.width, rc.y + baseLine)),
                                             clr, -1);
        };
        for (size_t i = 0; i < drawing_elements.rects_face.size(); ++i) {
            out_prims.emplace_back(rct(drawing_elements.rects_face[i], white_color));
            out_prims.emplace_back(pad(drawing_elements.labels_face[i], drawing_elements.rects_face[i], white_color));
            out_prims.emplace_back(txt(drawing_elements.labels_face[i], drawing_elements.rects_face[i], red_color));
        }
        for (size_t i = 0; i < drawing_elements.rects_det.size(); ++i) {
            out_prims.emplace_back(rct(drawing_elements.rects_det[i], white_color));
            out_prims.emplace_back(txt(drawing_elements.labels_det[i], drawing_elements.rects_det[i], red_color));
        }
    }
};

GAPI_OCV_KERNEL_ST(OCVTopAction, custom::TopAction, TopKState) {
    static void setup(const cv::GMatDesc&,
                      const cv::GArrayDesc&,
                      const ConstantParams&,
                      std::shared_ptr<TopKState>& top_k_st,
                      const cv::GCompileArgs& compileArgs) {
        top_k_st = std::make_shared<TopKState>();
    }
    static void run(const cv::Mat& in,
                    const TrackedObjects& tracked_actions,
                    const ConstantParams& params,
                    DrawingElements& drawing_elements,
                    cv::Mat& top_k,
                    TopKState& top_k_st) {
        if (static_cast<int>(top_k_st.top_k_obj_ids.size()) < params.top_flag) {
            for (const auto& action : tracked_actions) {
                if (action.label == params.top_action_id && top_k_st.top_k_obj_ids.count(action.object_id) == 0) {
                    const int action_id_in_top = top_k_st.top_k_obj_ids.size();
                    top_k_st.top_k_obj_ids.emplace(action.object_id, action_id_in_top);

                cv::Rect roi = action.rect;
                if (params.draw_ptr->rect_scale_x_ != 1 || params.draw_ptr->rect_scale_y_ != 1) {
                    roi.x = cvRound(roi.x * params.draw_ptr->rect_scale_x_);
                    roi.y = cvRound(roi.y * params.draw_ptr->rect_scale_y_);

                    roi.height = cvRound(roi.height * params.draw_ptr->rect_scale_y_);
                    roi.width = cvRound(roi.width * params.draw_ptr->rect_scale_x_);
                    }

                    roi.x = std::max(0, roi.x);
                    roi.y = std::max(0, roi.y);
                    roi.width = std::min(roi.width, in.cols - roi.x);
                    roi.height = std::min(roi.height, in.rows - roi.y);

                    auto frame_crop = in(roi).clone();
                    cv::resize(frame_crop, frame_crop,
                        cv::Size(params.draw_ptr->crop_width_, params.draw_ptr->crop_height_));
                    const int shift = (action_id_in_top + 1) * params.draw_ptr->margin_size_
                        + action_id_in_top * params.draw_ptr->crop_width_;
                    frame_crop.copyTo(top_k(cv::Rect(shift, params.draw_ptr->header_size_,
                                                            params.draw_ptr->crop_width_,
                                                            params.draw_ptr->crop_height_)));
                    if (static_cast<int>(top_k_st.top_k_obj_ids.size()) >= params.top_flag) {
                        break;
                    }
                }
            }
        }
        std::vector<cv::Rect> out_rects_det;
        std::vector<std::string> out_labels_det;
        for (const auto& action : tracked_actions) {
            auto box_color = white_color;
            std::string box_caption = "";

            if (top_k_st.top_k_obj_ids.count(action.object_id) > 0) {
                box_color = red_color;
                box_caption = std::to_string(top_k_st.top_k_obj_ids[action.object_id] + 1);
            }
            out_labels_det.emplace_back(box_caption);
            out_rects_det.emplace_back(action.rect);
        }
        drawing_elements = {out_rects_det, {}, out_labels_det, {}};
    }
};
// clang-format on

cv::gapi::GKernelPackage custom::kernels() {
    return cv::gapi::kernels<OCVFaceDetectorPostProc,
                             OCVPersonDetActionRecPostProc,
                             OCVAlignFacesForReidentification,
                             OCVGetActionTopHandsDetectionResult,
                             OCVGetRecognitionResult,
                             OCVBoxesAndLabels,
                             OCVRecognizeResultPostProc,
                             OCVGetRectFromImage,
                             OCVTopAction>();
}
