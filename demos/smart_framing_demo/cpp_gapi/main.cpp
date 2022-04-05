// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <chrono>
#include <string>
#include <vector>

#include <gflags/gflags.h>
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
#include <utils/default_flags.hpp>
#include <models/detection_model.h>



//#include "smart_framing_demo_gapi.hpp"
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

namespace {
constexpr char h_msg[] = "show the help message and exit";
DEFINE_bool(h, false, h_msg);

constexpr char i_msg[] =
"an input to process. The input must be a single image, a folder of images, video file or camera id. Default is 0";
DEFINE_string(i, "0", i_msg);

constexpr char camera_resolution_msg[] = "set camera resolution in format WxH.";
DEFINE_string(res, "1280x720", camera_resolution_msg);

constexpr char yolo_m_msg[] = "path to an .xml file with a trained YOLO v4 Tiny model.";
DEFINE_string(m_yolo, "", yolo_m_msg);

constexpr char sr_at_msg[] = "if Super Resolution (SR) model path provided, defines which SR architecture should be used. "
"Architecture type: Super Resolution - Default 3 channels input (3ch) or 1 channel input (1ch).";
DEFINE_string(at_sr, "3ch", sr_at_msg);

constexpr char sr_m_msg[] = "path to an .xml file with a trained Super Resolution Post Processing model.";
DEFINE_string(m_sr, "", sr_m_msg);

constexpr char kernel_package_msg[] = "G-API kernel package type: opencv, fluid (by default opencv is used).";
DEFINE_string(kernel_package, "opencv", kernel_package_msg);

constexpr char yolo_d_msg[] = "target device for YOLO v4 Tiny network (the list of available devices is shown below). "
"The demo will look for a suitable plugin for a specified device. Default value is \"CPU\".";
DEFINE_string(d_yolo, "CPU", yolo_d_msg);

constexpr char sr_d_msg[] = "target device for Super Resolution network (the list of available devices is shown below). "
"The demo will look for a suitable plugin for a specified device. Default value is \"CPU\".";
DEFINE_string(d_sr, "CPU", sr_d_msg);

constexpr char thr_conf_yolov4_msg[] = "YOLO v4 Tiny confidence threshold.";
DEFINE_double(t_conf_yolo, 0.5, thr_conf_yolov4_msg);

constexpr char thr_box_iou_yolov4_msg[] = "YOLO v4 Tiny box IOU threshold.";
DEFINE_double(t_box_iou_yolo, 0.5, thr_box_iou_yolov4_msg);

constexpr char advanced_pp_msg[] = "use advanced post-processing for the YOLO v4 Tiny.";
DEFINE_bool(advanced_pp, true, advanced_pp_msg);

constexpr char nireq_message[] = "number of infer requests. If this option is omitted, number of infer requests is determined automatically.";
DEFINE_uint32(nireq, 1, nireq_msg);

constexpr char num_threads_msg[] = "number of threads.";
DEFINE_uint32(nthreads, 0, num_threads_msg);

constexpr char num_streams_message[] = "number of streams to use for inference on the CPU or/and GPU in "
"throughput mode (for HETERO and MULTI device cases use format "
"<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>)";
DEFINE_string(nstreams, "", num_streams_msg);

constexpr char labels_msg[] = "path to a file with labels mapping.";
DEFINE_string(labels, "", labels_msg);

constexpr char crop_with_borders_msg[] = "apply alternative Smart Cropping with monocrhome borders.";
DEFINE_bool(crop_with_borders, false, crop_with_borders_msg);

constexpr char show_msg[] = "(don't) show output";
DEFINE_bool(show, true, show_msg);

constexpr char o_msg[] = "name of the output file(s) to save";
DEFINE_string(o, "", o_msg);

constexpr char lim_msg[] = "number of frames to store in output. If 0 is set, all frames are stored. Default is 1000";
DEFINE_uint32(lim, 1000, lim_msg);

// TODO: Make this option valid for single image case
constexpr char loop_msg[] = "enable playing video on a loop";
DEFINE_bool(loop, false, loop_msg);

constexpr char u_msg[] = "resource utilization graphs. Default is cdm. "
"c - average CPU load, d - load distribution over cores, m - memory usage, h - hide";
DEFINE_string(u, "cdm", u_msg);

void parse(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    if (FLAGS_h || 1 == argc) {
        std::cout << "\t[ -h]                                           " << h_msg
                  << "\n\t[--help]                                      print help on all arguments"
                  << "\n\t[ -i <INPUT>]                                 " << i_msg
                  << "\n\t[--res <WxH>]                                 " << camera_resolution_msg
                  << "\n\t --m_yolo <MODEL FILE>                        " << yolo_m_msg
                  << "\n\t[--at_sr]                                     " << sr_at_msg
                  << "\n\t[--m_sr <MODEL FILE>]                         " << sr_m_msg
                  << "\n\t[--kernel_package]                            " << kernel_package_msg
                  << "\n\t[--d_yolo <DEVICE>]                           " << yolo_d_msg
                  << "\n\t[--d_sr <DEVICE>]                             " << sr_d_msg
                  << "\n\t[--t_conf_yolo <NUMBER>]                      " << thr_conf_yolov4_msg
                  << "\n\t[--t_box_iou_yolo <NUMBER>]                   " << thr_box_iou_yolov4_msg
                  << "\n\t[--advanced_pp]                               " << advanced_pp_msg
                  << "\n\t[--nireq <NUMBER>]                            " << nireq_msg
                  << "\n\t[--nthreads <NUMBER>]                         " << num_threads_msg
                  << "\n\t[--nstreams <NUMBER>]                         " << num_streams_msg
                  << "\n\t[--lim <NUMBER>]                              " << lim_msg
                  << "\n\t[--loop]                                      " << loop_msg
                  << "\n\t[ -o <OUTPUT>]                                " << o_msg
                  << "\n\t[--labels]                                    " << labels_msg
                  << "\n\t[--crop_with_borders]                         " << crop_with_borders_msg
                  << "\n\t[--show] ([--noshow])                         " << show_msg
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
    if (FLAGS_m_yolo.empty()) {
        throw std::invalid_argument{"-m_yolo <MODEL FILE> can't be empty"};
    }
    /** Get OpenVINO runtime version **/
    slog::info << ov::get_openvino_version() << slog::endl;
}
}  // namespace

using GMat2 = std::tuple<cv::GMat, cv::GMat>;

int main(int argc, char *argv[]) {
    try {
        PerformanceMetrics metrics;

        // ---------- Parsing and validating of input arguments ----------
        //if (!util::ParseAndCheckCommandLine(argc, argv)) {
        //    return 0;
        //}
        parse(argc, argv);

        bool apply_sr = false;
        bool use_single_channel_sr = false;
        if (!FLAGS_m_sr.empty())
            apply_sr = true;
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
        if (apply_sr) {
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
        if (FLAGS_crop_with_borders) {
            out = custom::GSmartFramingMakeBorderKernel::on(in, yolo_detections);
        } else {
            out = custom::GSmartFramingKernel::on(in, yolo_detections);
        }
        if (apply_sr) {
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


        cv::GStreamingCompiled pipeline = cv::GComputation(cv::GIn(in, labels), cv::GOut(cv::gapi::copy(in), yolo_detections, out, apply_sr ? out_sr_pp : out))
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
        pipeline.setSource(cv::gin(cv::gapi::wip::make_src<custom::CommonCapSrc>(cap), coco_labels));

        cv::Size graphSize{ static_cast<int>(frame_size.width / 4), 60 };
        Presenter presenter(FLAGS_u, frame_size.height - graphSize.height - 10, graphSize);

        LazyVideoWriter videoWriter{ FLAGS_o, cap->fps(), FLAGS_lim };

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
            if (FLAGS_show) {
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
                if (apply_sr) {
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
