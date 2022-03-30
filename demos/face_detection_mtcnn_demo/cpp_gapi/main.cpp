// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stddef.h>  // for size_t

#include <algorithm>  // for copy
#include <chrono>  // for steady_clock
#include <exception>  // for exception
#include <iomanip>  // for operator<<, _Setprecision, setprecision, fixed
#include <limits>  // for numeric_limits
#include <memory>  // for shared_ptr, __shared_ptr_access, allocator_traits<>::value_type
#include <stdexcept>  // for logic_error
#include <string>  // for operator+, string
#include <tuple>  // for tie, tuple
#include <utility>  // for move
#include <vector>  // for vector

#include <opencv2/core.hpp>  // for Size, Mat, Rect, CV_Assert
#include <opencv2/gapi/core.hpp>  // for size, transpose
#include <opencv2/gapi/garg.hpp>  // for gout
#include <opencv2/gapi/garray.hpp>  // for GArray
#include <opencv2/gapi/gcommon.hpp>  // for compile_args, operator+=
#include <opencv2/gapi/gcomputation.hpp>  // for GComputation
#include <opencv2/gapi/gmat.hpp>  // for GMat
#include <opencv2/gapi/gopaque.hpp>  // for GOpaque
#include <opencv2/gapi/gproto.hpp>  // for GIn, GOut
#include <opencv2/gapi/gstreaming.hpp>  // for GStreamingCompiled, queue_capacity
#include <opencv2/gapi/imgproc.hpp>  // for BGR2RGB
#include <opencv2/gapi/infer.hpp>  // for infer, networks, operator+=, GNetworkType, G_API_NET, GNetPackage
#include <opencv2/gapi/infer/ie.hpp>  // for Params
#include <opencv2/gapi/render/render.hpp>  // for render3ch
#include <opencv2/highgui.hpp>  // for imshow, waitKey
#include <opencv2/imgproc.hpp>  // for FONT_HERSHEY_COMPLEX
#include <openvino/openvino.hpp>  // for get_openvino_version

#include <monitors/presenter.h>  // for Presenter
#include <utils/args_helper.hpp>  // for stringToSize
#include <utils/common.hpp>  // for fileNameNoExt, operator<<, showAvailableDevices
#include <utils/images_capture.h>  // for openImagesCapture, ImagesCapture, read_type, read_type::safe
#include <utils/ocv_common.hpp>  // for LazyVideoWriter
#include <utils/performance_metrics.hpp>  // for PerformanceMetrics, PerformanceMetrics::FPS, PerformanceMetrics::M...
#include <utils/slog.hpp>  // for LogStream, endl, info, err
#include <utils_gapi/stream_source.hpp>  // for CommonCapSrc

#include "custom_kernels.hpp"  // for RunNMS, ApplyRegression, Face, BBoxesToSquares, kernels, BuildFaces
#include "face_detection_mtcnn_demo.hpp"  // for FLAGS_th, FLAGS_i, FLAGS_m_o, FLAGS_m_p, FLAGS_m_r, FLAGS_loop
#include "gflags/gflags.h"  // for clstring, ParseCommandLineNonHelpFlags
#include "utils.hpp"  // for get_pnet_level_name, run_mtcnn_p, calculate_half_scales, calculate...

const int MAX_PYRAMID_LEVELS = 13;

namespace util {
bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    /** ---------- Parsing and validating input arguments ----------**/
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }
    if (FLAGS_i.empty())
        throw std::logic_error("Parameter -i is not set");
    if (FLAGS_m_p.empty())
        throw std::logic_error("Parameter -m_p is not set");
    if (FLAGS_m_r.empty())
        throw std::logic_error("Parameter -m_r is not set");
    if (FLAGS_m_o.empty())
        throw std::logic_error("Parameter -m_o is not set");
    return true;
}
}  // namespace util

namespace nets {
G_API_NET(MTCNNRefinement, <custom::GMat2(cv::GMat)>, "custom.mtcnn_refinement");
G_API_NET(MTCNNOutput, <custom::GMat3(cv::GMat)>, "custom.mtcnn_output");
}  // namespace nets

int main(int argc, char* argv[]) {
    try {
        PerformanceMetrics metrics;
        /** Get OpenVINO runtime version **/
        slog::info << ov::get_openvino_version() << slog::endl;

        if (!util::ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        /** Get information about frame **/
        std::shared_ptr<ImagesCapture> cap = openImagesCapture(FLAGS_i,
                                                               FLAGS_loop,
                                                               read_type::safe,
                                                               0,
                                                               std::numeric_limits<size_t>::max(),
                                                               stringToSize(FLAGS_res));
        const auto tmp = cap->read();
        cap.reset();
        cv::Size frame_size = cv::Size{tmp.cols, tmp.rows};
        cap = openImagesCapture(FLAGS_i,
                                FLAGS_loop,
                                read_type::safe,
                                0,
                                std::numeric_limits<size_t>::max(),
                                stringToSize(FLAGS_res));

        /** Calculate scales, number of pyramid levels and sizes for PNet pyramid **/
        std::vector<cv::Size> level_size;
        std::vector<double> scales;

        const auto pyramid_levels = FLAGS_hs ? calculate_half_scales(frame_size, scales, level_size)
                                             : calculate_scales(frame_size, scales, level_size);
        CV_Assert(pyramid_levels <= MAX_PYRAMID_LEVELS);

        /** ---------------- Main graph of demo ---------------- **/
        /** Graph input
         * Proposal part of MTCNN graph
         * Preprocessing BGR2RGB + transpose (NCWH is expected instead of NCHW)
         **/
        cv::GMat in_original;
        cv::GMat in_originalRGB = cv::gapi::BGR2RGB(in_original);
        cv::GMat in_transposedRGB = cv::gapi::transpose(in_originalRGB);
        cv::GOpaque<cv::Size> in_sz = cv::gapi::streaming::size(in_original);
        cv::GMat regressions[MAX_PYRAMID_LEVELS];
        cv::GMat scores[MAX_PYRAMID_LEVELS];
        cv::GArray<custom::Face> nms_p_faces[MAX_PYRAMID_LEVELS];
        cv::GArray<custom::Face> total_faces[MAX_PYRAMID_LEVELS];

        /** The very first PNet pyramid layer to init total_faces[0] **/
        std::tie(regressions[0], scores[0]) = run_mtcnn_p(in_transposedRGB, get_pnet_level_name(level_size[0]));
        cv::GArray<custom::Face> faces0 = custom::BuildFaces::on(scores[0],
                                                                 regressions[0],
                                                                 static_cast<float>(scales[0]),
                                                                 static_cast<float>(FLAGS_th));
        cv::GArray<custom::Face> final_p_faces_for_bb2squares = custom::ApplyRegression::on(faces0, true);
        cv::GArray<custom::Face> final_faces_pnet0 = custom::BBoxesToSquares::on(final_p_faces_for_bb2squares);
        total_faces[0] = custom::RunNMS::on(final_faces_pnet0, 0.5f, false);

        /** The rest PNet pyramid layers to accumlate all layers result in total_faces[PYRAMID_LEVELS - 1]] **/
        for (int i = 1; i < pyramid_levels; ++i) {
            std::tie(regressions[i], scores[i]) = run_mtcnn_p(in_transposedRGB, get_pnet_level_name(level_size[i]));
            cv::GArray<custom::Face> faces = custom::BuildFaces::on(scores[i],
                                                                    regressions[i],
                                                                    static_cast<float>(scales[i]),
                                                                    static_cast<float>(FLAGS_th));
            cv::GArray<custom::Face> final_p_faces_for_bb2squares_i = custom::ApplyRegression::on(faces, true);
            cv::GArray<custom::Face> final_faces_pnet_i = custom::BBoxesToSquares::on(final_p_faces_for_bb2squares_i);
            nms_p_faces[i] = custom::RunNMS::on(final_faces_pnet_i, 0.5f, false);
            total_faces[i] = custom::AccumulatePyramidOutputs::on(total_faces[i - 1], nms_p_faces[i]);
        }

        /** Proposal post-processing **/
        cv::GArray<custom::Face> final_faces_pnet = custom::RunNMS::on(total_faces[pyramid_levels - 1], 0.7f, true);

        /** Refinement part of MTCNN graph **/
        cv::GArray<cv::Rect> faces_roi_pnet = custom::R_O_NetPreProcGetROIs::on(final_faces_pnet, in_sz);
        cv::GArray<cv::GMat> regressionsRNet, scoresRNet;
        std::tie(regressionsRNet, scoresRNet) =
            cv::gapi::infer<nets::MTCNNRefinement>(faces_roi_pnet, in_transposedRGB);

        /** Refinement post-processing **/
        cv::GArray<custom::Face> rnet_post_proc_faces =
            custom::RNetPostProc::on(final_faces_pnet, scoresRNet, regressionsRNet, static_cast<float>(FLAGS_th));
        cv::GArray<custom::Face> nms07_r_faces_total = custom::RunNMS::on(rnet_post_proc_faces, 0.7f, false);
        cv::GArray<custom::Face> final_r_faces_for_bb2squares = custom::ApplyRegression::on(nms07_r_faces_total, true);
        cv::GArray<custom::Face> final_faces_rnet = custom::BBoxesToSquares::on(final_r_faces_for_bb2squares);

        /** Output part of MTCNN graph **/
        cv::GArray<cv::Rect> faces_roi_rnet = custom::R_O_NetPreProcGetROIs::on(final_faces_rnet, in_sz);
        cv::GArray<cv::GMat> regressionsONet, scoresONet, landmarksONet;
        std::tie(regressionsONet, landmarksONet, scoresONet) =
            cv::gapi::infer<nets::MTCNNOutput>(faces_roi_rnet, in_transposedRGB);

        /** Output post-processing **/
        cv::GArray<custom::Face> onet_post_proc_faces = custom::ONetPostProc::on(final_faces_rnet,
                                                                                 scoresONet,
                                                                                 regressionsONet,
                                                                                 landmarksONet,
                                                                                 static_cast<float>(FLAGS_th));
        cv::GArray<custom::Face> final_o_faces_for_nms07 = custom::ApplyRegression::on(onet_post_proc_faces, true);
        cv::GArray<custom::Face> nms07_o_faces_total = custom::RunNMS::on(final_o_faces_for_nms07, 0.7f, true);
        cv::GArray<custom::Face> final_faces_onet = custom::SwapFaces::on(nms07_o_faces_total);

        /** Draw ROI and marks **/
        auto rendered =
            cv::gapi::wip::draw::render3ch(in_original, custom::BoxesAndMarks::on(in_original, final_faces_onet));

        cv::GComputation graph_mtcnn(cv::GIn(in_original), cv::GOut(rendered));
        /** ---------------- End of graph ---------------- **/
        /** Configure networks **/

        // MTCNN Refinement detection network
        auto mtcnnr_net =
            cv::gapi::ie::Params<nets::MTCNNRefinement>{
                FLAGS_m_r,  // path to topology IR
                fileNameNoExt(FLAGS_m_r) + ".bin",  // path to weights
                FLAGS_d_r,  // device specifier
            }
                .cfgOutputLayers({"conv5-2", "prob1"})
                .cfgInputLayers({"data"});

        // MTCNN Output detection network
        auto mtcnno_net =
            cv::gapi::ie::Params<nets::MTCNNOutput>{
                FLAGS_m_o,  // path to topology IR
                fileNameNoExt(FLAGS_m_o) + ".bin",  // path to weights
                FLAGS_d_o,  // device specifier
            }
                .cfgOutputLayers({"conv6-2", "conv6-3", "prob1"})
                .cfgInputLayers({"data"});

        auto networks_mtcnn = cv::gapi::networks(mtcnnr_net, mtcnno_net);

        // MTCNN Proposal detection network
        for (int i = 0; i < pyramid_levels; ++i) {
            std::string net_id = get_pnet_level_name(level_size[i]);
            std::vector<size_t> reshape_dims = {1, 3, size_t(level_size[i].width), size_t(level_size[i].height)};
            cv::gapi::ie::Params<cv::gapi::Generic> mtcnnp_net{
                net_id,  // tag
                FLAGS_m_p,  // path to topology IR
                fileNameNoExt(FLAGS_m_p) + ".bin",  // path to weights
                FLAGS_d_p,  // device specifier
            };
            mtcnnp_net.cfgInputReshape("data", reshape_dims);
            networks_mtcnn += cv::gapi::networks(mtcnnp_net);
        }

        /** Custom kernels **/
        auto kernels_mtcnn = custom::kernels();
        auto mtcnn_args = cv::compile_args(networks_mtcnn, kernels_mtcnn);
        if (FLAGS_qc != 0) {
            mtcnn_args += cv::compile_args(cv::gapi::streaming::queue_capacity{FLAGS_qc});
        }
        auto pipeline_mtcnn = graph_mtcnn.compileStreaming(std::move(mtcnn_args));

        /** ---------------- The execution part ---------------- **/
        pipeline_mtcnn.setSource<custom::CommonCapSrc>(cap);

        cv::Size graphSize{static_cast<int>(frame_size.width / 4), 60};
        Presenter presenter(FLAGS_u, frame_size.height - graphSize.height - 10, graphSize);

        LazyVideoWriter videoWriter{FLAGS_o, cap->fps(), FLAGS_limit};

        /** Output Mat for result **/
        cv::Mat out_image;
        bool isStart = true;
        const auto startTime = std::chrono::steady_clock::now();
        pipeline_mtcnn.start();
        while (pipeline_mtcnn.pull(cv::gout(out_image))) {
            if (isStart) {
                metrics.update(startTime,
                               out_image,
                               {10, 22},
                               cv::FONT_HERSHEY_COMPLEX,
                               0.65,
                               {200, 10, 10},
                               2,
                               PerformanceMetrics::MetricTypes::FPS);
                isStart = false;
            } else {
                metrics.update({},
                               out_image,
                               {10, 22},
                               cv::FONT_HERSHEY_COMPLEX,
                               0.65,
                               {200, 10, 10},
                               2,
                               PerformanceMetrics::MetricTypes::FPS);
            }
            videoWriter.write(out_image);
            if (!FLAGS_no_show) {
                cv::imshow("Face detection mtcnn demo G-API", out_image);
                int key = cv::waitKey(1);
                /** Press 'Esc' or 'Q' to quit **/
                if (key == 27)
                    break;
                if (key == 81)  // Q
                    break;
                else
                    presenter.handleKey(key);
            }
        }

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
