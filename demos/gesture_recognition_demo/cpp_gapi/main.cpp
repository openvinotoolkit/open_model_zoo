// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stddef.h>  // for size_t

#include <algorithm>  // for copy
#include <chrono>  // for steady_clock
#include <exception>  // for exception
#include <iomanip>  // for operator<<, _Setprecision, setprecision, fixed
#include <limits>  // for numeric_limits
#include <memory>  // for shared_ptr, make_shared, __shared_ptr_access
#include <stdexcept>  // for logic_error
#include <string>  // for operator+, string
#include <utility>  // for move
#include <vector>  // for vector

#include <opencv2/core.hpp>  // for Size, Mat, Scalar_, Vec
#include <opencv2/gapi/garg.hpp>  // for gin, gout
#include <opencv2/gapi/garray.hpp>  // for GArray
#include <opencv2/gapi/gcommon.hpp>  // for compile_args
#include <opencv2/gapi/gcomputation.hpp>  // for GComputation
#include <opencv2/gapi/gmat.hpp>  // for GMat
#include <opencv2/gapi/gopaque.hpp>  // for GOpaque
#include <opencv2/gapi/gproto.hpp>  // for GIn, GOut
#include <opencv2/gapi/gstreaming.hpp>  // for GStreamingCompiled
#include <opencv2/gapi/infer.hpp>  // for infer, infer2, networks, GNetworkType, G_API_NET
#include <opencv2/gapi/infer/ie.hpp>  // for Params
#include <opencv2/gapi/streaming/source.hpp>  // for make_src
#include <opencv2/highgui.hpp>  // for waitKey
#include <opencv2/imgproc.hpp>  // for FONT_HERSHEY_COMPLEX

#include <monitors/presenter.h>  // for Presenter
#include <utils/args_helper.hpp>  // for stringToSize
#include <utils/common.hpp>  // for fileNameNoExt, showAvailableDevices
#include <utils/images_capture.h>  // for openImagesCapture, ImagesCapture, read_type, read_type::safe
#include <utils/ocv_common.hpp>  // for LazyVideoWriter
#include <utils/performance_metrics.hpp>  // for PerformanceMetrics, PerformanceMetrics::FPS, PerformanceMetrics...
#include <utils/slog.hpp>  // for LogStream, endl, info, err

#include "custom_kernels.hpp"  // for kernels, ConstructClip, ExtractBoundingBox, GestureRecognitionP...
#include "gesture_recognition_demo_gapi.hpp"  // for FLAGS_m_a, FLAGS_m_d, FLAGS_i, FLAGS_c, FLAGS_d_a, FLAGS_d_d
#include "gflags/gflags.h"  // for clstring, ParseCommandLineNonHelpFlags
#include "stream_source.hpp"  // for GestRecCapSource
#include "tracker.hpp"  // for TrackedObject (ptr only), TrackedObjects
#include "utils.hpp"  // for getNetShape, fill_labels
#include "visualizer.hpp"  // for Visualizer

bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    /** ---------- Parsing and validating input arguments ----------**/
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;
    if (FLAGS_i.empty())
        throw std::logic_error("Parameter -i is not set");
    if (FLAGS_m_a.empty())
        throw std::logic_error("Parameter -m_a is not set");
    if (FLAGS_m_d.empty())
        throw std::logic_error("Parameter -m_d is not set");
    if (FLAGS_c.empty())
        throw std::logic_error("Parameter -c is not set");

    return true;
}

namespace nets {
G_API_NET(PersonDetection, <cv::GMat(cv::GMat)>, "person_detection");
G_API_NET(ActionRecognition, <cv::GMat(cv::GMat)>, "action_recognition");
}  // namespace nets

int main(int argc, char* argv[]) {
    try {
        PerformanceMetrics metrics;
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        /** Get information about nets **/
        const auto pd_net_shape = getNetShape(FLAGS_m_d);
        const auto ar_net_shape = getNetShape(FLAGS_m_a);

        /** Get information about frame from cv::VideoCapture **/
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

        /** Share runtime id with graph **/
        auto current_person_id_m = std::make_shared<size_t>(0);

        /** ---------------- Main graph of demo ---------------- **/
        /** Graph inputs **/
        cv::GArray<cv::GMat> batch;
        cv::GOpaque<std::shared_ptr<size_t>> current_person_id;

        cv::GMat fast_frame = custom::GetFastFrame::on(batch, frame_size);

        /** Person detection **/
        cv::GMat detections = cv::gapi::infer<nets::PersonDetection>(fast_frame);

        /** Get ROIs from detections **/
        cv::GArray<TrackedObject> objects = custom::ExtractBoundingBox::on(detections, fast_frame, pd_net_shape);

        /** Track detection **/
        cv::GArray<TrackedObject> tracked = custom::TrackPerson::on(fast_frame, objects);

        /** Create clip for AR net **/
        cv::GArray<cv::GMat> clip =
            custom::ConstructClip::on(batch, tracked, ar_net_shape, frame_size, current_person_id);

        /** Action recognition **/
        cv::GArray<cv::GMat> actions = cv::gapi::infer2<nets::ActionRecognition>(fast_frame, clip);

        /** Get action label **/
        cv::GOpaque<int> label = custom::GestureRecognitionPostprocessing::on(actions, static_cast<float>(FLAGS_t));

        /** Inputs and outputs of graph **/
        auto graph = cv::GComputation(cv::GIn(batch, current_person_id), cv::GOut(fast_frame, tracked, label));
        /** ---------------- End of graph ---------------- **/
        /** Configure networks **/
        auto person_detection =
            cv::gapi::ie::Params<nets::PersonDetection>{
                FLAGS_m_d,  // path to model
                fileNameNoExt(FLAGS_m_d) + ".bin",  // path to weights
                FLAGS_d_d  // device to use
            }
                .cfgOutputLayers({"boxes"});  // This clarification here because of
                                              // GAPI take the first layer name from OutputsInfo
                                              // for one output G_API_NET API
        slog::info << "The Person Detection ASL model " << FLAGS_m_d << " is loaded to " << FLAGS_d_d << " device."
                   << slog::endl;

        auto action_recognition =
            cv::gapi::ie::Params<nets::ActionRecognition>{
                FLAGS_m_a,  // path to model
                fileNameNoExt(FLAGS_m_a) + ".bin",  // path to weights
                FLAGS_d_a  // device to use
            }
                .cfgOutputLayers({"output"});  // This clarification here because of
                                               // GAPI take the first layer name from OutputsInfo
                                               // for one output G_API_NET API
        slog::info << "The Action Recognition model " << FLAGS_m_a << " is loaded to " << FLAGS_d_a << " device."
                   << slog::endl;

        /** Custom kernels **/
        auto kernels = custom::kernels();
        auto networks = cv::gapi::networks(person_detection, action_recognition);
        auto comp = cv::compile_args(kernels, networks);
        auto pipeline = graph.compileStreaming(std::move(comp));

        /** Output containers for results **/
        cv::Mat out_frame;
        TrackedObjects out_detections;
        int out_label_number;

        /** ---------------- The execution part ---------------- **/
        const float batch_constant_FPS = 15;
        auto drop_batch = std::make_shared<bool>(false);
        pipeline.setSource(cv::gin(cv::gapi::wip::make_src<custom::GestRecCapSource>(cap,
                                                                                     frame_size,
                                                                                     static_cast<int>(ar_net_shape[1]),
                                                                                     batch_constant_FPS,
                                                                                     drop_batch),
                                   current_person_id_m));

        std::string gestureWindowName = "Gesture";

        cv::Size graphSize{static_cast<int>(frame_size.width / 4), 60};
        Presenter presenter(FLAGS_u, frame_size.height - graphSize.height - 10, graphSize);

        LazyVideoWriter videoWriter{FLAGS_o, cap->fps(), FLAGS_limit};

        /** Fill labels container from file with classes **/
        const auto labels = fill_labels(FLAGS_c);
        size_t current_id = 0;
        size_t last_id = current_id;
        int gesture = 0;

        /** Configure drawing utilities **/
        Visualizer visualizer(FLAGS_no_show, gestureWindowName, labels, FLAGS_s);

        bool isStart = true;
        const auto startTime = std::chrono::steady_clock::now();
        pipeline.start();
        while (pipeline.pull(cv::gout(out_frame, out_detections, out_label_number))) {
            /** Put FPS to frame**/
            if (isStart) {
                metrics.update(startTime,
                               out_frame,
                               {10, 22},
                               cv::FONT_HERSHEY_COMPLEX,
                               0.65,
                               {200, 10, 10},
                               2,
                               PerformanceMetrics::MetricTypes::FPS);
                isStart = false;
            } else {
                metrics.update({},
                               out_frame,
                               {10, 22},
                               cv::FONT_HERSHEY_COMPLEX,
                               0.65,
                               {200, 10, 10},
                               2,
                               PerformanceMetrics::MetricTypes::FPS);
            }

            /** Display system parameters **/
            presenter.drawGraphs(out_frame);
            /** Display the results **/
            visualizer.show(out_frame, out_detections, out_label_number, current_id, gesture);
            gesture = 0;

            videoWriter.write(out_frame);

            /** Controls **/
            int key = cv::waitKey(1);
            if (key == 0x1B)
                break;  // (esc button) exit
            else if (key >= 48 && key <= 57)
                current_id = key - 48;  // buttons for person id
            else if (key == 0x0D)
                out_label_number = -1;  // (Enter) reset last gesture
            else if (key == 'f')
                gesture = 1;  // next gesture
            else if (key == 'b')
                gesture = -1;  // prev gesture
            else
                presenter.handleKey(key);

            /** Share id with graph **/
            if (current_id < out_detections.size()) {
                *drop_batch = last_id != current_id;
                *current_person_id_m = current_id;
                last_id = current_id;
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
    slog::info << "Execution successful" << slog::endl;

    return 0;
}
