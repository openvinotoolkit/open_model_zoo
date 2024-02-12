// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stddef.h>

#include <algorithm>
#include <chrono>
#include <exception>
#include <iomanip>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gflags/gflags.h>
#include <opencv2/core.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/garray.hpp>
#include <opencv2/gapi/gcommon.hpp>
#include <opencv2/gapi/gcomputation.hpp>
#include <opencv2/gapi/gmat.hpp>
#include <opencv2/gapi/gopaque.hpp>
#include <opencv2/gapi/gproto.hpp>
#include <opencv2/gapi/gscalar.hpp>
#include <opencv2/gapi/gstreaming.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/infer/parsers.hpp>
#include <opencv2/gapi/render/render.hpp>
#include <opencv2/gapi/streaming/format.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <monitors/presenter.h>
#include <utils/common.hpp>
#include <utils/images_capture.h>
#include <utils/ocv_common.hpp>
#include <utils/performance_metrics.hpp>
#include <utils/slog.hpp>
#include <utils_gapi/stream_source.hpp>

#include "action_detector.hpp"
#include "actions.hpp"
#include "custom_kernels.hpp"
#include "detector.hpp"
#include "drawing_helper.hpp"
#include "initialize.hpp"
#include "kernel_packages.hpp"
#include "recognizer.hpp"
#include "smart_classroom_demo_gapi.hpp"
#include "tracker.hpp"

namespace util {
bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }
    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }
    if (FLAGS_m_act.empty() && FLAGS_m_fd.empty()) {
        throw std::logic_error("At least one parameter -m_act or -m_fd must be set");
    }
    return true;
}
}  // namespace util

namespace config {
/** The function packs demo's flags that received from command line to appropriate structures
 * for convenience of work **/
std::tuple<NetsFlagsPack, ArgsFlagsPack, preparation::FGFlagsPack> packDemoFlags() {
    using namespace preparation;
    return std::make_tuple(NetsFlagsPack{FLAGS_m_act,
                                         FLAGS_m_fd,
                                         FLAGS_m_lm,
                                         FLAGS_m_reid,
                                         FLAGS_d_act,
                                         FLAGS_d_fd,
                                         FLAGS_d_lm,
                                         FLAGS_d_reid,
                                         FLAGS_inh_fd,
                                         FLAGS_inw_fd},
                           ArgsFlagsPack{FLAGS_teacher_id,
                                         FLAGS_min_ad,
                                         FLAGS_d_ad,
                                         FLAGS_student_ac,
                                         FLAGS_top_ac,
                                         FLAGS_teacher_ac,
                                         FLAGS_top_id,
                                         FLAGS_a_top,
                                         FLAGS_ss_t,
                                         FLAGS_no_show},
                           FGFlagsPack{FLAGS_greedy_reid_matching,
                                       FLAGS_t_reid,
                                       FLAGS_fg,
                                       FLAGS_crop_gallery,
                                       FLAGS_t_reg_fd,
                                       FLAGS_min_size_fr,
                                       FLAGS_exp_r_fd});
}
}  // namespace config

int main(int argc, char* argv[]) {
    try {
        PerformanceMetrics metrics;

        /** This demo covers 4 certain topologies and cannot be generalized **/
        if (!util::ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        /** Prepare parameters **/
        const std::string video_path     = FLAGS_i;
        const auto ad_model_path         = FLAGS_m_act;
        const auto fd_model_path         = FLAGS_m_fd;
        const auto fr_model_path         = FLAGS_m_reid;
        const auto lm_model_path         = FLAGS_m_lm;

        /** Pack demo flags to appropriate unions **/
        config::NetsFlagsPack netsFlags;
        config::ArgsFlagsPack configFlags;
        preparation::FGFlagsPack fgFlags;
        std::tie(netsFlags, configFlags, fgFlags) = config::packDemoFlags();

        /** Print info about demo's properties **/
        config::printInfo(netsFlags, FLAGS_teacher_id, FLAGS_top_id);

        /** Set input source (image, video, camera) **/
        std::shared_ptr<ImagesCapture> cap =
            openImagesCapture(FLAGS_i, FLAGS_loop, read_type::safe, 0, FLAGS_read_limit);
        cv::Mat tmp = cap->read();
        cap.reset();
        cv::Size frame_size = cv::Size{tmp.cols, tmp.rows};
        cap = openImagesCapture(FLAGS_i, FLAGS_loop, read_type::safe, 0, FLAGS_read_limit);
        /** Fill shared constants and trackers parameters **/
        TrackerParams tracker_reid_params, tracker_action_params;
        ConstantParams const_params;
        std::tie(const_params, tracker_reid_params, tracker_action_params) =
            config::getGraphArgs(video_path, frame_size, cap->fps(), configFlags);

        /** Create net's package **/
        cv::gapi::GNetPackage networks;

        /** Configure nets **/
        cv::Size act_net_in_size;
        cv::Scalar reid_net_in_size;
        config::configNets(netsFlags, networks, act_net_in_size, reid_net_in_size);

        /** Configure and create action detector **/
        std::shared_ptr<ActionDetection> act_det_ptr;
        if (!ad_model_path.empty()) {
            act_det_ptr = config::createActDetPtr(ad_model_path,
                                                  act_net_in_size,
                                                  const_params.actions_map.size(),
                                                  FLAGS_t_ad,
                                                  FLAGS_t_ar);
        }

        /** Configure and create face detector **/
        std::shared_ptr<detection::FaceDetection> face_det_ptr;
        if (!fd_model_path.empty()) {
            face_det_ptr = std::make_shared<detection::FaceDetection>(config::getDetConfig(FLAGS_exp_r_fd));
        }

        /** Find identities metric for each face from gallery **/
        std::shared_ptr<FaceRecognizer> face_rec_ptr;
        std::vector<std::string> face_id_to_label_map;
        if (!fd_model_path.empty() && !fr_model_path.empty() && !lm_model_path.empty()) {
            face_rec_ptr = preparation::processingFaceGallery(networks, fgFlags, reid_net_in_size);
            face_id_to_label_map = face_rec_ptr->GetIDToLabelMap();
        } else {
            slog::warn << "Face recognition models are disabled!" << slog::endl;
            if (const_params.actions_type == TEACHER) {
                slog::err << "Face recognition must be enabled to recognize teacher actions." << slog::endl;
                return 1;
            }
        }
        if (const_params.actions_type == TEACHER && !face_rec_ptr->LabelExists(const_params.teacher_id)) {
            slog::err << "Teacher id does not exist in the gallery!" << slog::endl;
            return 1;
        }
        if (!FLAGS_ad.empty()) {
            if (const_params.actions_type != STUDENT) {
                slog::err << "-ad requires -teacher_id and -a_top to be unset" << slog::endl;
                return 1;
            }
            if (FLAGS_fg.empty()) {
                slog::err << "-ad requires -fg to be set" << slog::endl;
                return 1;
            }
        }

        /** ---------------- Main graph of demo ---------------- **/
        cv::GMat in;
        cv::GMat frame = cv::gapi::copy(in);
        /** First graph output **/
        auto outs = GOut(frame);
        /** Initialize empty GArrays **/
        cv::GArray<cv::GMat> embeddings(std::vector<cv::Mat>{});
        cv::GArray<DetectedAction> persons_with_actions(std::vector<DetectedAction>{});
        cv::GArray<cv::Rect> rects(std::vector<cv::Rect>{});

        if (const_params.actions_type != TOP_K) {
            if (!fd_model_path.empty()) {
                /** Face detection **/
                cv::GMat detections = cv::gapi::infer<nets::FaceDetector>(in);
                cv::GOpaque<cv::Size> sz = cv::gapi::streaming::size(in);
                cv::GArray<cv::Rect> face_rects =
                    cv::gapi::parseSSD(detections, sz, static_cast<float>(FLAGS_t_fd), true, true);
                rects = custom::FaceDetectorPostProc::on(in, face_rects, face_det_ptr);
                if (!fr_model_path.empty() && !lm_model_path.empty()) {
                    /** Get landmarks **/
                    cv::GArray<cv::GMat> landmarks = cv::gapi::infer<nets::LandmarksDetector>(rects, in);
                    /** Get aligned faces **/
                    cv::GScalar net_size(reid_net_in_size);
                    cv::GArray<cv::GMat> align_faces =
                        custom::AlignFacesForReidentification::on(in, landmarks, rects, net_size);
                    /** Get face identities metrics for each person **/
                    embeddings = cv::gapi::infer2<nets::FaceReidentificator>(in, align_faces);
                }
            }
        }

        if (!ad_model_path.empty()) {
            cv::GMat location, detect_confidences, priorboxes, action_con1, action_con2, action_con3, action_con4;
            /** Action detection-recognition **/
            std::tie(location, detect_confidences, priorboxes, action_con1, action_con2, action_con3, action_con4) =
                cv::gapi::infer<nets::PersonDetActionRec>(in);

            /** Get actions for each person on frame **/
            persons_with_actions = custom::PersonDetActionRecPostProc::on(in,
                                                                          location,
                                                                          detect_confidences,
                                                                          priorboxes,
                                                                          action_con1,
                                                                          action_con2,
                                                                          action_con3,
                                                                          action_con4,
                                                                          act_det_ptr);
        }
        cv::GOpaque<DrawingElements> draw_elements;
        cv::GArray<TrackedObject> tracked_actions;
        if (const_params.actions_type != TOP_K) {
            /** Main demo scenario **/
            cv::GOpaque<FaceTrack> face_track;
            cv::GOpaque<size_t> work_num_frames;
            /** Recognize actions and faces **/
            std::tie(tracked_actions, face_track, work_num_frames) =
                custom::GetRecognitionResult::on(in,
                                                 rects,
                                                 persons_with_actions,
                                                 embeddings,
                                                 face_rec_ptr,
                                                 const_params);

            cv::GOpaque<std::string> stream_log, stat_log, det_log;
            cv::GArray<std::string> face_ids(face_id_to_label_map);
            /** Get roi and labels for drawing and set logs **/
            std::tie(draw_elements, stream_log, stat_log, det_log) =
                custom::RecognizeResultPostProc::on(in,
                                                    tracked_actions,
                                                    face_track,
                                                    face_ids,
                                                    work_num_frames,
                                                    const_params);
            /** Main demo part of graph output **/
            outs += GOut(work_num_frames, stream_log, stat_log, det_log);
        } else {
            /** Top action case **/
            cv::GMat top_k;
            /** Recognize actions **/
            tracked_actions = custom::GetActionTopHandsDetectionResult::on(in, persons_with_actions);
            /** Get roi and labels for drawing **/
            std::tie(draw_elements, top_k) = custom::TopAction::on(in, tracked_actions, const_params);
            /** Top action case part of graph output **/
            outs += GOut(top_k);
        }
        /** Draw ROI and labels **/
        auto rendered =
            cv::gapi::wip::draw::render3ch(frame, custom::BoxesAndLabels::on(frame, draw_elements, const_params));
        /** Last graph output is frame to draw **/
        outs += GOut(rendered);

        /** Pipeline's input and outputs**/
        cv::GComputation pipeline(cv::GIn(in), std::move(outs));

        cv::GStreamingCompiled stream =
            pipeline.compileStreaming(cv::compile_args(custom::kernels(),
                                                       networks,
                                                       TrackerParamsPack{tracker_reid_params, tracker_action_params},
                                                       FLAGS_r));

        /** ---------------- The execution part ---------------- **/
        stream.setSource<custom::CommonCapSrc>(cap);

        /** Service constants **/
        size_t work_num_frames = 0;
        const char SPACE_KEY = 32;
        const char ESC_KEY = 27;
        bool monitoring_enabled = const_params.actions_type == TOP_K ? false : true;
        cv::Size graphSize{static_cast<int>(frame_size.width / 4), 60};

        /** Presenter for rendering system parameters **/
        Presenter presenter(FLAGS_u, frame_size.height - graphSize.height - 10, graphSize);

        LazyVideoWriter videoWriter{FLAGS_o, cap->fps(), FLAGS_limit};

        /** Result containers associated with graph output **/
        cv::Mat out_frame, proc, top_k;
        std::string stream_log, stat_log, det_log;
        auto out_vector = cv::gout(out_frame);
        if (const_params.actions_type == TOP_K) {
            out_vector += cv::gout(top_k, proc);
        } else {
            out_vector += cv::gout(work_num_frames, stream_log, stat_log, det_log, proc);
        }

        /** TOP_K case starts without processing **/
        if (const_params.actions_type != TOP_K)
            stream.start();

        bool isStart = true;
        const auto startTime = std::chrono::steady_clock::now();
        /** Main cycle **/
        while (true) {
            char key = cv::waitKey(1);
            presenter.handleKey(key);
            if (key == ESC_KEY) {
                break;
            }

            if (const_params.actions_type == TOP_K) {
                if ((key == SPACE_KEY && !monitoring_enabled) || (key == SPACE_KEY && monitoring_enabled)) {
                    /** SPACE_KEY & monitoring_enabled trigger **/
                    monitoring_enabled = !monitoring_enabled;
                    const_params.draw_ptr->ClearTopWindow();
                }
            }
            if (monitoring_enabled) {
                if (!stream.running()) {
                    /** TOP_K part. SPACE_KEY is pressed, monitoring enabled
                     *  Compile and start graph **/
                    stream.setSource<custom::CommonCapSrc>(cap);
                    stream.start();
                }
                if (!stream.pull(std::move(out_vector))) {
                    /** Main part. Processing is always on **/
                    break;
                }
            } else {
                /** TOP_K part. monitoring isn't enabled **/
                if (stream.running())
                    stream.stop();
                /** Get clear frame **/
                out_frame = cap->read();
                if (!out_frame.data)
                    break;
                const auto new_height = cvRound(out_frame.rows * const_params.draw_ptr->rect_scale_y_);
                const auto new_width = cvRound(out_frame.cols * const_params.draw_ptr->rect_scale_x_);
                cv::resize(out_frame, out_frame, cv::Size(new_width, new_height));
                presenter.drawGraphs(out_frame);
                if (isStart) {
                    metrics.update(startTime,
                                   proc,
                                   {10, 22},
                                   cv::FONT_HERSHEY_COMPLEX,
                                   0.65,
                                   {200, 10, 10},
                                   2,
                                   PerformanceMetrics::MetricTypes::FPS);
                    isStart = false;
                } else {
                    metrics.update({},
                                   proc,
                                   {10, 22},
                                   cv::FONT_HERSHEY_COMPLEX,
                                   0.65,
                                   {200, 10, 10},
                                   2,
                                   PerformanceMetrics::MetricTypes::FPS);
                }

                const_params.draw_ptr->Show(out_frame);
                const_params.draw_ptr->ShowCrop();
            }
            if (const_params.actions_type == TOP_K && monitoring_enabled) {
                /** TOP_K part. monitoring is enabled and graph is started **/
                const_params.draw_ptr->Show(proc);
                const_params.draw_ptr->ShowCrop(top_k);
            } else if (const_params.actions_type != TOP_K) {
                /** Main part. Processing is always on **/
                presenter.drawGraphs(proc);
                if (isStart) {
                    metrics.update(startTime,
                                   proc,
                                   {10, 22},
                                   cv::FONT_HERSHEY_COMPLEX,
                                   0.65,
                                   {200, 10, 10},
                                   2,
                                   PerformanceMetrics::MetricTypes::FPS);
                    isStart = false;
                } else {
                    metrics.update({},
                                   proc,
                                   {10, 22},
                                   cv::FONT_HERSHEY_COMPLEX,
                                   0.65,
                                   {200, 10, 10},
                                   2,
                                   PerformanceMetrics::MetricTypes::FPS);
                }
                const_params.draw_ptr->Show(proc);
            }
            videoWriter.write(proc);
            /** Console log, if exists **/
            if (!stream_log.empty()) {
                slog::debug << stream_log << slog::endl;
            }
        }
        const_params.draw_ptr->Finalize();

        /** Print logs to files **/
        std::ofstream act_stat_log_stream, act_det_log_stream;
        if (!FLAGS_al.empty()) {
            act_det_log_stream.open(FLAGS_al, std::fstream::out);
            act_det_log_stream << "data"
                               << "[" << std::endl;
            act_det_log_stream << det_log << "]";
        }
        act_stat_log_stream.open(FLAGS_ad, std::fstream::out);
        act_stat_log_stream << stat_log << std::endl;

        slog::info << "Metrics report:" << slog::endl;
        slog::info << "\tFPS: " << std::fixed << std::setprecision(1) << metrics.getTotal().fps << slog::endl;
        slog::info << presenter.reportMeans() << slog::endl;
    } catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    } catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }
    return 0;
}
