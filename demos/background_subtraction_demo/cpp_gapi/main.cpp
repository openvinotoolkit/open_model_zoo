// Copyright (C) 2022-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stddef.h>

#include <algorithm>
#include <chrono>
#include <exception>
#include <iomanip>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gflags/gflags.h>
#include <opencv2/core.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/gcommon.hpp>
#include <opencv2/gapi/gcomputation.hpp>
#include <opencv2/gapi/gkernel.hpp>
#include <opencv2/gapi/gmat.hpp>
#include <opencv2/gapi/gproto.hpp>
#include <opencv2/gapi/gstreaming.hpp>
#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/own/assert.hpp>
#include <opencv2/gapi/streaming/source.hpp>
#include <opencv2/gapi/util/optional.hpp>
#include <opencv2/gapi/streaming/onevpl/source.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <openvino/openvino.hpp>

#include <monitors/presenter.h>
#include <utils/args_helper.hpp>
#include <utils/common.hpp>
#include <utils/config_factory.h>
#include <utils/images_capture.h>
#include <utils/ocv_common.hpp>
#include <utils/performance_metrics.hpp>
#include <utils/slog.hpp>
#include <utils_gapi/kernel_package.hpp>
#include <utils_gapi/stream_source.hpp>

#include "background_subtraction_demo_gapi.hpp"
#include "custom_kernels.hpp"

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
    if (FLAGS_m.empty())
        throw std::logic_error("Parameter -m is not set");
    if (FLAGS_at.empty()) {
        throw std::logic_error("Parameter -at is not set");
    }
    return true;
}

}  // namespace util

int main(int argc, char* argv[]) {
    try {
        PerformanceMetrics metrics;

        /** Get OpenVINO runtime version **/
        slog::info << ov::get_openvino_version() << slog::endl;
        // ---------- Parsing and validating of input arguments ----------
        if (!util::ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        /** ---------------- Main graph of demo ---------------- **/
        const bool is_blur = FLAGS_blur_bgr != 0;

        std::shared_ptr<custom::NNBGReplacer> model;
        if (FLAGS_at == "maskrcnn") {
            model = std::make_shared<custom::MaskRCNNBGReplacer>(FLAGS_m);
        } else if (FLAGS_at == "background-matting") {
            model = std::make_shared<custom::BGMattingReplacer>(FLAGS_m);
        } else {
            slog::err << "No model type or invalid model type (-at) provided: " + FLAGS_at << slog::endl;
            return -1;
        }

        /** Get information about frame **/
        std::shared_ptr<ImagesCapture> cap = openImagesCapture(FLAGS_i,
                                                               FLAGS_loop,
                                                               read_type::safe,
                                                               0,
                                                               std::numeric_limits<size_t>::max(),
                                                               stringToSize(FLAGS_res));
        const auto tmp = cap->read();
        cv::Size frame_size = cv::Size{tmp.cols, tmp.rows};
        cap.reset();
        // NB: oneVPL source rounds up frame size by 16
        // so size might be different from what ImagesCapture reads.
        if (FLAGS_use_onevpl) {
            frame_size.width  = cv::alignSize(frame_size.width, 16);
            frame_size.height = cv::alignSize(frame_size.height, 16);
        }

        cv::GComputation comp([&] {
            cv::GFrame in;
            cv::GMat bgr = cv::gapi::streaming::BGR(in);
            // NB: target_bgr is optional second input which implies a background
            // that will change user video background. If user don't specify
            // it and specifies --bgr_blur then second input won't be used since
            // it's needed only to blur user background, otherwise second input will be used.
            cv::optional<cv::GMat> target_bgr;

            cv::GMat bgr_resized;
            if (is_blur && FLAGS_target_bgr.empty()) {
                bgr_resized = bgr;
            } else {
                target_bgr = cv::util::make_optional<cv::GMat>(cv::GMat());
                bgr_resized = cv::gapi::resize(target_bgr.value(), frame_size);
            }

            auto background =
                is_blur ? cv::gapi::blur(bgr_resized, cv::Size(FLAGS_blur_bgr, FLAGS_blur_bgr)) : bgr_resized;

            auto result = model->replace(in, bgr, frame_size, background);

            auto graph_inputs = cv::GIn(in);
            if (target_bgr.has_value()) {
                graph_inputs += cv::GIn(target_bgr.value());
            }

            return cv::GComputation(std::move(graph_inputs), cv::GOut(result));
        });

        /** Configure network **/
        auto config = ConfigFactory::getUserConfig(FLAGS_d, FLAGS_nireq, FLAGS_nstreams, FLAGS_nthreads);
        // clang-format off
        const auto net =
            cv::gapi::ie::Params<cv::gapi::Generic>{
                model->getName(),
                FLAGS_m,  // path to topology IR
                fileNameNoExt(FLAGS_m) + ".bin",  // path to weights
                FLAGS_d  // device specifier
            }.cfgNumRequests(config.maxAsyncRequests)
             .pluginConfig(config.getLegacyConfig());
        // clang-format on

        slog::info << "The background matting model " << FLAGS_m << " is loaded to " << FLAGS_d << " device."
                   << slog::endl;

        auto kernels = cv::gapi::combine(custom::kernels(), util::getKernelPackage(FLAGS_kernel_package));
        auto pipeline = comp.compileStreaming(cv::compile_args(kernels, cv::gapi::networks(net)));

        /** Output container for result **/
        cv::Mat output;

        /** ---------------- The execution part ---------------- **/
        cap = openImagesCapture(FLAGS_i,
                                FLAGS_loop,
                                read_type::safe,
                                0,
                                std::numeric_limits<size_t>::max(),
                                stringToSize(FLAGS_res));
        cv::gapi::wip::IStreamSource::Ptr media_cap;
        if (FLAGS_use_onevpl) {
            auto onevpl_params = util::parseVPLParams(FLAGS_onevpl_params);
            if (FLAGS_onevpl_pool_size != 0) {
                onevpl_params.push_back(
                    cv::gapi::wip::onevpl::CfgParam::create_frames_pool_size(FLAGS_onevpl_pool_size));
            }
            media_cap = cv::gapi::wip::make_onevpl_src(FLAGS_i, std::move(onevpl_params));
        } else {
            media_cap = cv::gapi::wip::make_src<custom::MediaCommonCapSrc>(cap);
        }

        auto pipeline_inputs = cv::gin(std::move(media_cap));
        if (!is_blur && FLAGS_target_bgr.empty()) {
            cv::Scalar default_color(155, 255, 120);
            pipeline_inputs += cv::gin(cv::Mat(frame_size, CV_8UC3, default_color));
        } else if (!FLAGS_target_bgr.empty()) {
            std::shared_ptr<ImagesCapture> target_bgr_cap =
                openImagesCapture(FLAGS_target_bgr, true, read_type::safe, 0, std::numeric_limits<size_t>::max());
            pipeline_inputs += cv::gin(cv::gapi::wip::make_src<custom::CommonCapSrc>(target_bgr_cap));
        }

        pipeline.setSource(std::move(pipeline_inputs));
        std::string windowName = "Background subtraction demo G-API";
        int delay = 1;

        cv::Size graphSize{static_cast<int>(frame_size.width / 4), 60};
        Presenter presenter(FLAGS_u, frame_size.height - graphSize.height - 10, graphSize);

        LazyVideoWriter videoWriter{FLAGS_o, cap->fps(), FLAGS_limit};

        bool isStart = true;
        const auto startTime = std::chrono::steady_clock::now();
        pipeline.start();

        while (pipeline.pull(cv::gout(output))) {
            presenter.drawGraphs(output);
            if (isStart) {
                metrics.update(startTime,
                               output,
                               {10, 22},
                               cv::FONT_HERSHEY_COMPLEX,
                               0.65,
                               {200, 10, 10},
                               2,
                               PerformanceMetrics::MetricTypes::FPS);
                isStart = false;
            } else {
                metrics.update({},
                               output,
                               {10, 22},
                               cv::FONT_HERSHEY_COMPLEX,
                               0.65,
                               {200, 10, 10},
                               2,
                               PerformanceMetrics::MetricTypes::FPS);
            }

            videoWriter.write(output);

            if (!FLAGS_no_show) {
                cv::imshow(windowName, output);
                int key = cv::waitKey(delay);
                /** Press 'Esc' to quit **/
                if (key == 27) {
                    break;
                } else {
                    presenter.handleKey(key);
                }
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
