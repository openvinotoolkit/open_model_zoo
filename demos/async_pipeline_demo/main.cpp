/*
// Copyright (C) 2018-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

/**
* \brief The entry point for the Inference Engine object_detection_demo_ssd_async demo application
* \file object_detection_demo_ssd_async/main.cpp
* \example object_detection_demo_ssd_async/main.cpp
*/

#include <iostream>
#include <vector>
#include <string>

#include <monitors/presenter.h>
#include <samples/ocv_common.hpp>
#include <samples/args_helper.hpp>
#include <samples/slog.hpp>

#include "detection_pipeline.h"
#include "segmentation_pipeline.h"
#include "config_factory.h"

#include <gflags/gflags.h>
#include <iostream>

#include <unordered_map>

static const char help_message[] = "Print a usage message.";
static const char video_message[] = "Required. Path to a video file (specify \"cam\" to work with camera).";
static const char model_message[] = "Required. Path to an .xml file with a trained model.";
static const char target_device_message[] = "Optional. Specify the target device to infer on (the list of available devices is shown below). "
"Default value is CPU. Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. "
"The demo will look for a suitable plugin for a specified device.";
static const char labels_message[] = "Optional. Path to a file with labels mapping.";
static const char performance_counter_message[] = "Optional. Enables per-layer performance report.";
static const char custom_cldnn_message[] = "Required for GPU custom kernels. "
"Absolute path to the .xml file with the kernel descriptions.";
static const char custom_cpu_library_message[] = "Required for CPU custom layers. "
"Absolute path to a shared library with the kernel implementations.";
static const char thresh_output_message[] = "Optional. Probability threshold for detections.";
static const char raw_output_message[] = "Optional. Inference results as raw values.";
static const char input_resizable_message[] = "Optional. Enables resizable input with support of ROI crop & auto resize.";
static const char num_inf_req_message[] = "Optional. Number of infer requests.";
static const char num_threads_message[] = "Optional. Number of threads.";
static const char num_streams_message[] = "Optional. Number of streams to use for inference on the CPU or/and GPU in "
"throughput mode (for HETERO and MULTI device cases use format "
"<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>)";
static const char loop_input_message[] = "Optional. Iterate over input infinitely.";
static const char no_show_processed_video[] = "Optional. Do not show processed video.";
static const char utilization_monitors_message[] = "Optional. List of monitors to show initially.";
static const char usecase_message[] = "Type of model: d-detection, s-segmentation.";

DEFINE_bool(h, false, help_message);
DEFINE_string(i, "", video_message);
DEFINE_string(m, "", model_message);
DEFINE_string(d, "CPU", target_device_message);
DEFINE_string(labels, "", labels_message);
DEFINE_bool(pc, false, performance_counter_message);
DEFINE_string(c, "", custom_cldnn_message);
DEFINE_string(l, "", custom_cpu_library_message);
DEFINE_bool(r, false, raw_output_message);
DEFINE_double(t, 0.5, thresh_output_message);
DEFINE_bool(auto_resize, false, input_resizable_message);
DEFINE_uint32(nireq, 2, num_inf_req_message);
DEFINE_uint32(nthreads, 0, num_threads_message);
DEFINE_string(nstreams, "", num_streams_message);
DEFINE_bool(loop_input, false, loop_input_message);
DEFINE_bool(no_show, false, no_show_processed_video);
DEFINE_string(u, "", utilization_monitors_message);
DEFINE_string(use, "", usecase_message);

/**
* \brief This function shows a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "object_detection_demo_ssd_async [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                        " << help_message << std::endl;
    std::cout << "    -i \"<path>\"               " << video_message << std::endl;
    std::cout << "    -m \"<path>\"               " << model_message << std::endl;
    std::cout << "      -l \"<absolute_path>\"    " << custom_cpu_library_message << std::endl;
    std::cout << "          Or" << std::endl;
    std::cout << "      -c \"<absolute_path>\"    " << custom_cldnn_message << std::endl;
    std::cout << "    -d \"<device>\"             " << target_device_message << std::endl;
    std::cout << "    -_fil \"<path>\"          " << labels_message << std::endl;
    std::cout << "    -pc                       " << performance_counter_message << std::endl;
    std::cout << "    -r                        " << raw_output_message << std::endl;
    std::cout << "    -t                        " << thresh_output_message << std::endl;
    std::cout << "    -auto_resize              " << input_resizable_message << std::endl;
    std::cout << "    -nireq \"<integer>\"        " << num_inf_req_message << std::endl;
    std::cout << "    -nthreads \"<integer>\"     " << num_threads_message << std::endl;
    std::cout << "    -nstreams                 " << num_streams_message << std::endl;
    std::cout << "    -loop_input               " << loop_input_message << std::endl;
    std::cout << "    -no_show                  " << no_show_processed_video << std::endl;
    std::cout << "    -u                        " << utilization_monitors_message << std::endl;
    std::cout << "    -use                      " << usecase_message << std::endl;
}

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    return true;
}

void paintInfo(cv::Mat& frame, const PipelineBase::PerformanceInfo& info) {
    std::ostringstream out;

    out.str("");
    out << "FPS: " << std::fixed << std::setprecision(1) << info.FPS;
    putHighlightedText(frame, out.str(), cv::Point2f(10, 30), cv::FONT_HERSHEY_TRIPLEX, 0.75, cv::Scalar(10, 200, 10), 2);

    out.str("");
    out << "Avg Latency: " << std::fixed << std::setprecision(1) <<
        (std::chrono::duration_cast<std::chrono::milliseconds>(info.latencySum).count() / info.framesCount)
        << " ms";
    putHighlightedText(frame, out.str(), cv::Point2f(10, 60), cv::FONT_HERSHEY_TRIPLEX, 0.75, cv::Scalar(200, 10, 10), 2);
    out.str("");
    out << "Pool: " << std::fixed << std::setprecision(1) <<
        info.numRequestsInUse << "/" << FLAGS_nireq;
    putHighlightedText(frame, out.str(), cv::Point2f(10, 90), cv::FONT_HERSHEY_TRIPLEX, 0.75, cv::Scalar(200, 10, 10), 2);

}

int main(int argc, char *argv[]) {
    try {
        /** This demo covers certain topology and cannot be generalized for any object detection **/
        std::cout << "InferenceEngine: " << *InferenceEngine::GetInferenceEngineVersion() << std::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        //------------------------------- Preparing Input ------------------------------------------------------
        slog::info << "Reading input" << slog::endl;
        cv::VideoCapture cap;

        if (!((FLAGS_i == "cam") ? cap.open(0) : cap.open(FLAGS_i.c_str()))) {
            throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
        }

        // read input (video) frame
        cv::Mat curr_frame;
        cap >> curr_frame;

        if (!cap.grab()) {
            throw std::logic_error("This demo supports only video (or camera) inputs !!! "
                "Failed getting next frame from the " + FLAGS_i);
        }

        //------------------------------ Creating pipeline object ------------------------------------------------
        std::unique_ptr<PipelineBase> pipeline;
        if (FLAGS_use == "s") {
            pipeline.reset(new SegmentationPipeline(FLAGS_m, ConfigFactory::GetUserConfig()));
        }
        else if (FLAGS_use == "d") {
            auto labels = FLAGS_labels.empty() ?
                std::vector<std::string>() :
                DetectionPipeline::loadLabels(FLAGS_labels);

            pipeline.reset(new DetectionPipeline(
                FLAGS_m, ConfigFactory::GetUserConfig(), (float)FLAGS_t, FLAGS_auto_resize, labels));
        }
        else {
            printf("\nERROR: Use case mode (-use) is not defined or invalid. Exiting...\n");
            return -1;
        }

        //------------------------------ Running Detection routines ----------------------------------------------
        Presenter presenter;

        auto startTimePoint = std::chrono::steady_clock::now();
        while (true) {
            //--- Capturing frame. If previous frame hasn't been inferred yet, reuse it instead of capturing new one
            int64_t frameNum;
            if (curr_frame.empty())
                cap >> curr_frame;
            frameNum = pipeline->submitImage(curr_frame);
            if (frameNum >= 0)
                curr_frame.release();

            //--- Checking for results and rendering data if it's ready
            //--- If you need just plain data without rendering - check for getProcessedResult() function
            cv::Mat outFrame = pipeline->obtainAndRenderData();

            //--- Showing results and device information
            if (!outFrame.empty() && !FLAGS_no_show) {
                presenter.drawGraphs(outFrame);
                paintInfo(outFrame, pipeline->getPerformanceInfo());
                cv::imshow("Inference Result", outFrame);
            }

            //--- Waiting for free input slot or output data available. Function will return immediately if any of them are available.
            pipeline->waitForData();

            //--- Processing keyboard events
            const int key = cv::waitKey(1);

            if (!FLAGS_no_show) {
                if (27 == key || 'q' == key || 'Q' == key) {  // Esc
                    break;
                }
                else {
                    presenter.handleKey(key);
                }
            }
        }

        //// --------------------------- Report metrics -------------------------------------------------------
        pipeline->waitForTotalCompletion(); // If some requests are still executing, let's wait for them to get full statistics
        auto info = pipeline->getPerformanceInfo();
        slog::info << slog::endl << "Metric reports:" << slog::endl;

        std::cout << std::endl << "Total time: "
            << std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::steady_clock::now() - startTimePoint).count()
            << " ms" << std::endl;

        std::cout << "Avg Latency: " << std::fixed << std::setprecision(1) <<
            (std::chrono::duration_cast<std::chrono::milliseconds>(info.latencySum).count() / info.framesCount)
            << " ms" << std::endl;

        std::cout << "FPS: " << std::fixed << std::setprecision(1) << info.FPS << std::endl;

        std::cout << presenter.reportMeans() << std::endl;
    }
    catch (const std::exception& error) {
        std::cerr << "[ ERROR ] " << error.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
        return 1;
    }

    slog::info << slog::endl << "The execution has completed successfully" << slog::endl;
    return 0;
}
