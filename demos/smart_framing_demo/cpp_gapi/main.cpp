// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <chrono>
#include <string>
#include <vector>

#include <opencv2/gapi/streaming/cap.hpp>
#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/gapi/core.hpp>

#include <opencv2/gapi/fluid/core.hpp>
#include <opencv2/gapi/fluid/imgproc.hpp>
#include <opencv2/gapi/cpu/core.hpp>
#include <opencv2/gapi/cpu/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <openvino/openvino.hpp>

#include <monitors/presenter.h>
#include <utils_gapi/stream_source.hpp>
#include <utils/args_helper.hpp>
#include <utils/config_factory.h>
#include <utils/images_capture.h>
#include <utils/ocv_common.hpp>
#include <utils/performance_metrics.hpp>


#include "smart_framing_demo_gapi.hpp"
#include "custom_kernels.hpp"


namespace util {
bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    /** ---------- Parsing and validating input arguments ----------**/
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }
    if (FLAGS_i.empty())
        throw std::logic_error("Parameter -i is not set");
    if (FLAGS_m_yolo.empty())
        throw std::logic_error("Parameter -m_yolo is not set");
    if (FLAGS_m_sr.empty() && FLAGS_apply_sr)
        throw std::logic_error("Parameter -m_sr is not set");
    if (FLAGS_at_sr.empty() && FLAGS_apply_sr) {
        throw std::logic_error("Parameter -at_sr is not set");
    }
    return true;
}

static cv::gapi::GKernelPackage getKernelPackage(const std::string& type) {
    if (type == "opencv") {
        return cv::gapi::combine(cv::gapi::core::cpu::kernels(),
                                 cv::gapi::imgproc::cpu::kernels());
    } else if (type == "fluid") {
        return cv::gapi::combine(cv::gapi::core::fluid::kernels(),
                                 cv::gapi::imgproc::fluid::kernels());
    } else {
        throw std::logic_error("Unsupported kernel package type: " + type);
    }
    GAPI_Assert(false && "Unreachable code!");
}

} // namespace util

using GMat2 = std::tuple<cv::GMat, cv::GMat>;

int main(int argc, char *argv[]) {
    try {
        PerformanceMetrics metrics;

        /** Get OpenVINO runtime version **/
        slog::info << ov::get_openvino_version() << slog::endl;
        // ---------- Parsing and validating of input arguments ----------
        if (!util::ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        bool use_single_channel_sr = false;
        if (FLAGS_apply_sr) {
            if (FLAGS_at_sr == "1ch") {
                use_single_channel_sr = true;
            } else if (FLAGS_at_sr == "3ch") {
                use_single_channel_sr = false;
            } else {
                slog::err << "No model type or invalid model type (-at_sr) provided: " + FLAGS_at_sr << slog::endl;
                return -1;
            }
        }

        std::vector<std::string> coco_labels;
        if (!FLAGS_labels.empty()) {
            coco_labels = DetectionModel::loadLabels(FLAGS_labels);
        }


        /** ---------------- Main graph of demo ---------------- **/
        cv::gapi::GNetPackage networks;

        /** Configure networks **/
        G_API_NET(YOLOv4TinyNet, <GMat2(cv::GMat)>, "yolov4tiny_detector");
        const auto net = cv::gapi::ie::Params<YOLOv4TinyNet>{
            FLAGS_m_yolo,                         // path to topology IR
            fileNameNoExt(FLAGS_m_yolo) + ".bin", // path to weights
            FLAGS_d_yolo
        }.cfgOutputLayers({ "conv2d_20/BiasAdd/Add", "conv2d_17/BiasAdd/Add" }).cfgInputLayers({ "image_input" });
        G_API_NET(SRNet, <cv::GMat(cv::GMat)>, "super_resolution");
        G_API_NET(SRNet3ch, <cv::GMat(cv::GMat, cv::GMat)>, "super_resolution_3ch");
        if (FLAGS_apply_sr) {
            using sr_net = cv::gapi::ie::Params<SRNet>;
            using sr_net3ch = cv::gapi::ie::Params<SRNet3ch>;
            auto sr = sr_net{
                FLAGS_m_sr,                         // path to topology IR
                fileNameNoExt(FLAGS_m_sr) + ".bin", // path to weights
                FLAGS_d_sr
            };
            auto sr3ch = sr_net3ch{
                FLAGS_m_sr,                         // path to topology IR
                fileNameNoExt(FLAGS_m_sr) + ".bin", // path to weights
                FLAGS_d_sr
            }.cfgInputLayers({ "1", "0" });

            if (use_single_channel_sr) {
                networks = cv::gapi::networks(net, sr);
            } else {
                networks = cv::gapi::networks(net, sr3ch);
            }
        } else {
            networks = cv::gapi::networks(net);
        }

        cv::GMat blob26x26; //float32[1,255,26,26]
        cv::GMat blob13x13; //float32[1,255,13,13]
        cv::GArray<custom::DetectedObject> yolo_detections;
        cv::GArray<std::string> labels;

        // Now build the graph
        cv::GMat in;
        cv::GMat out; //cropped resized output image
        cv::GMat out_sr; //cropped resized output image after super resolution
        cv::GMat out_sr_pp; //cropped resized output image after super resolution post processing

        std::tie(blob26x26, blob13x13) = cv::gapi::infer<YOLOv4TinyNet>(in);
        yolo_detections = custom::GYOLOv4TinyPostProcessingKernel::on(in, blob26x26, blob13x13, labels, FLAGS_t_conf_yolo, FLAGS_t_box_iou_yolo, FLAGS_advanced_pp);
        out = custom::GSmartFramingKernel::on(in, yolo_detections);
        if (FLAGS_apply_sr) {
            if (use_single_channel_sr) {
                cv::GMat b, g, r;
                std::tie(b, g, r) = cv::gapi::split3(out);
                auto out_b = custom::GCvt32Fto8U::on(cv::gapi::infer<SRNet>(b));
                auto out_g = custom::GCvt32Fto8U::on(cv::gapi::infer<SRNet>(g));
                auto out_r = custom::GCvt32Fto8U::on(cv::gapi::infer<SRNet>(r));
                out_sr_pp = cv::gapi::merge3(out_b, out_g, out_r);
            } else {
                out_sr = cv::gapi::infer<SRNet3ch>(out, out);
                out_sr_pp = custom::GSuperResolutionPostProcessingKernel::on(out_sr);
            }
        }

        /** Custom kernels plus CPU or Fluid **/
        auto kernels = cv::gapi::combine(custom::kernels(),
            util::getKernelPackage(FLAGS_kernel_package));


        cv::GStreamingCompiled pipeline = cv::GComputation(cv::GIn(in, labels), cv::GOut(cv::gapi::copy(in), yolo_detections, out, FLAGS_apply_sr ? out_sr_pp : out))
                                                           .compileStreaming(cv::compile_args(kernels, networks));

        /** ---------------- End of graph ---------------- **/

        /** Get information about frame **/
        std::shared_ptr<ImagesCapture> cap = openImagesCapture(FLAGS_i, FLAGS_loop, read_type::safe, 0,
            std::numeric_limits<size_t>::max(), stringToSize(FLAGS_res));
        const auto tmp = cap->read();
        cap.reset();
        cv::Size frame_size = cv::Size{ tmp.cols, tmp.rows };
        cap = openImagesCapture(FLAGS_i, FLAGS_loop, read_type::safe, 0,
            std::numeric_limits<size_t>::max(), stringToSize(FLAGS_res));


        /** ---------------- The execution part ---------------- **/
        //pipeline.setSource<custom::CommonCapSrc>(cap);
        //pipeline.setSource<custom::CommonCapSrc>(cap);
        pipeline.setSource(cv::gin(cv::gapi::wip::make_src<custom::CommonCapSrc>(cap), coco_labels));

        cv::Size graphSize{ static_cast<int>(frame_size.width / 4), 60 };
        Presenter presenter(FLAGS_u, frame_size.height - graphSize.height - 10, graphSize);

        LazyVideoWriter videoWriter{ FLAGS_o, cap->fps(), FLAGS_limit };

        /** Output Mat for result **/
        cv::Mat image;
        cv::Mat out_image;
        cv::Mat out_image_sr;
        std::vector<custom::DetectedObject> objects;


        bool isStart = true;
        const auto startTime = std::chrono::steady_clock::now();
        pipeline.start();
        while (pipeline.pull(cv::gout(image, objects, out_image, out_image_sr))) {
            if (isStart) {
                metrics.update(startTime, out_image_sr, { 10, 22 }, cv::FONT_HERSHEY_COMPLEX,
                    0.65, { 200, 10, 10 }, 2, PerformanceMetrics::MetricTypes::FPS);
                isStart = false;
            } else {
                metrics.update({}, out_image_sr, { 10, 22 }, cv::FONT_HERSHEY_COMPLEX,
                    0.65, { 200, 10, 10 }, 2, PerformanceMetrics::MetricTypes::FPS);
            }
            videoWriter.write(out_image_sr);
            if (!FLAGS_no_show) {
                //Draw detections on original image
                for (const auto& el : objects) {
                    slog::debug << el << slog::endl;
                    cv::rectangle(image, el, cv::Scalar{ 0,255,0 }, 2, cv::LINE_8, 0);
                    cv::putText(image, el.label, el.tl(), cv::FONT_HERSHEY_SIMPLEX, 1.0, { 0,255,0 }, 2);
                }
                //////////////////////////////////////////////////
                ////Smart framing prototype
                cv::Rect init_rect;
                for (const auto& el : objects) {
                    if (el.labelID == 0) {//person ID
                        init_rect = init_rect | static_cast<cv::Rect>(el);
                    }
                }
                cv::rectangle(image, init_rect, cv::Scalar{ 255,0,0 }, 3, cv::LINE_8, 0);
                cv::imshow("Smart framing demo G-API person detections", image);
                cv::imshow("Smart framing demo G-API smart framing result", out_image);
                if (FLAGS_apply_sr) {
                    cv::imshow("Smart framing demo G-API smart framing with SR result", out_image_sr);
                }
                int key = cv::waitKey(1);
                /** Press 'Esc' or 'Q' to quit **/
                if (key == 27)
                    break;
                if (key == 81) // Q
                    break;
                else
                    presenter.handleKey(key);
            }
        }

        slog::info << "Metrics report:" << slog::endl;
        slog::info << "\tFPS: " << std::fixed << std::setprecision(1) << metrics.getTotal().fps << slog::endl;
        slog::info << presenter.reportMeans() << slog::endl;
    }
    catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }
    return 0;
}
