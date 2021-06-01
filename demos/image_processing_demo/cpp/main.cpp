/*
// Copyright (C) 2021 Intel Corporation
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
* \brief The entry point for the Inference Engine image_processing_demo_async demo application
* \file image_processing_async/main.cpp
* \example image_processing_async/main.cpp
*/

#include <iostream>
#include <map>
#include <string>

#include <monitors/presenter.h>
#include <utils/ocv_common.hpp>
#include <utils/args_helper.hpp>
#include <utils/slog.hpp>
#include <utils/images_capture.h>
#include <utils/default_flags.hpp>
#include <utils/performance_metrics.hpp>
#include <unordered_map>
#include <gflags/gflags.h>
#include <sys/stat.h>

#include <pipelines/async_pipeline.h>
#include <models/super_resolution_model.h>
#include <models/deblurring_model.h>
#include <pipelines/metadata.h>
#include "visualizer.hpp"

DEFINE_INPUT_FLAGS
DEFINE_OUTPUT_FLAGS

static const char help_message[] = "Print a usage message.";
static const char at_message[] = "Required. Type of the network, either 'sr' for Super Resolution task or 'deblur' for Deblurring";
static const char model_message[] = "Required. Path to an .xml file with a trained model.";
static const char target_device_message[] = "Optional. Specify the target device to infer on (the list of available devices is shown below). "
"Default value is CPU. Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. "
"The demo will look for a suitable plugin for a specified device.";
static const char performance_counter_message[] = "Optional. Enables per-layer performance report.";
static const char custom_cldnn_message[] = "Required for GPU custom kernels. "
"Absolute path to the .xml file with the kernel descriptions.";
static const char custom_cpu_library_message[] = "Required for CPU custom layers. "
"Absolute path to a shared library with the kernel implementations.";
static const char nireq_message[] = "Optional. Number of infer requests. If this option is omitted, number of infer requests is determined automatically.";
static const char num_threads_message[] = "Optional. Number of threads.";
static const char num_streams_message[] = "Optional. Number of streams to use for inference on the CPU or/and GPU in "
"throughput mode (for HETERO and MULTI device cases use format "
"<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>)";
static const char no_show_processed_video[] = "Optional. Do not show processed video.";
static const char utilization_monitors_message[] = "Optional. List of monitors to show initially.";
static const char output_resolution_message[] = "Optional. Specify the maximum output window resolution "
    "in (width x height) format. Example: 1280x720. Input frame size used by default.";

DEFINE_bool(h, false, help_message);
DEFINE_string(at, "", at_message);
DEFINE_string(m, "", model_message);
DEFINE_string(d, "CPU", target_device_message);
DEFINE_bool(pc, false, performance_counter_message);
DEFINE_string(c, "", custom_cldnn_message);
DEFINE_string(l, "", custom_cpu_library_message);
DEFINE_uint32(nireq, 0, nireq_message);
DEFINE_uint32(nthreads, 0, num_threads_message);
DEFINE_string(nstreams, "", num_streams_message);
DEFINE_bool(no_show, false, no_show_processed_video);
DEFINE_string(u, "", utilization_monitors_message);
DEFINE_string(output_resolution, "", output_resolution_message);

/**
* \brief This function shows a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "image_processing_demo_async [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                        " << help_message << std::endl;
    std::cout << "    -at \"<type>\"              " << at_message << std::endl;
    std::cout << "    -i \"<path>\"               " << input_message << std::endl;
    std::cout << "    -m \"<path>\"               " << model_message << std::endl;
    std::cout << "    -o \"<path>\"               " << output_message << std::endl;
    std::cout << "    -limit \"<num>\"            " << limit_message << std::endl;
    std::cout << "      -l \"<absolute_path>\"    " << custom_cpu_library_message << std::endl;
    std::cout << "          Or" << std::endl;
    std::cout << "      -c \"<absolute_path>\"    " << custom_cldnn_message << std::endl;
    std::cout << "    -d \"<device>\"             " << target_device_message << std::endl;
    std::cout << "    -pc                       " << performance_counter_message << std::endl;
    std::cout << "    -nireq \"<integer>\"        " << nireq_message << std::endl;
    std::cout << "    -nthreads \"<integer>\"     " << num_threads_message << std::endl;
    std::cout << "    -nstreams                 " << num_streams_message << std::endl;
    std::cout << "    -loop                     " << loop_message << std::endl;
    std::cout << "    -no_show                  " << no_show_processed_video << std::endl;
    std::cout << "    -output_resolution        " << output_resolution_message << std::endl;
    std::cout << "    -u                        " << utilization_monitors_message << std::endl;
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

    if (FLAGS_at.empty()) {
        throw std::logic_error("Parameter -at is not set");
    }

    if (!FLAGS_output_resolution.empty() && FLAGS_output_resolution.find("x") == std::string::npos) {
        throw std::logic_error("Correct format of -output_resolution parameter is \"width\"x\"height\".");
    }
    return true;
}

std::unique_ptr<ImageModel> getModel(const cv::Size& frameSize, const std::string& type) {
    if (type == "sr") {
        return std::unique_ptr<ImageModel>(new SuperResolutionModel(FLAGS_m, frameSize));
    }
    if (type == "deblur") {
        return std::unique_ptr<ImageModel>(new DeblurringModel(FLAGS_m, frameSize));
    }
    throw std::invalid_argument("No model type or invalid model type (-at) provided: " + FLAGS_at);
}

int main(int argc, char *argv[]) {
    try {
        PerformanceMetrics metrics;

        slog::info << "InferenceEngine: " << printable(*InferenceEngine::GetInferenceEngineVersion()) << slog::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        //------------------------------- Preparing Input ------------------------------------------------------
        slog::info << "Reading input" << slog::endl;
        auto cap = openImagesCapture(FLAGS_i, FLAGS_loop);
        cv::Mat curr_frame;

        auto startTime = std::chrono::steady_clock::now();
        curr_frame = cap->read();
        if (curr_frame.empty()) {
            throw std::logic_error("Can't read an image from the input");
        }

        //------------------------------ Running ImageProcessing routines ----------------------------------------------
        InferenceEngine::Core core;
        std::unique_ptr<ImageModel> model = getModel(cv::Size(curr_frame.cols, curr_frame.rows), FLAGS_at);
        AsyncPipeline pipeline(std::move(model),
            ConfigFactory::getUserConfig(FLAGS_d,FLAGS_l,FLAGS_c,FLAGS_pc,FLAGS_nireq,FLAGS_nstreams,FLAGS_nthreads),
            core);
        Presenter presenter(FLAGS_u);

        int64_t frameNum = pipeline.submitData(ImageInputData(curr_frame),
            std::make_shared<ImageMetaData>(curr_frame, startTime));;

        bool keepRunning = true;
        std::unique_ptr<ResultBase> result;
        uint32_t framesProcessed = 0;
        cv::VideoWriter videoWriter;

        cv::Size outputResolution;
        OutputTransform outputTransform = OutputTransform();
        size_t found = FLAGS_output_resolution.find("x");
        Visualizer view(FLAGS_at);

        // interactive mode for single image
        if (cap->getType() == "IMAGE" && !FLAGS_loop && !FLAGS_no_show) {
            pipeline.waitForTotalCompletion();
            result = pipeline.getResult();
            if (found == std::string::npos) {
                outputResolution = result->asRef<ImageResult>().resultImage.size();
                cv::Size viewSize = view.getSize();
                if (outputResolution.height > viewSize.height || outputResolution.width > viewSize.width)
                    outputResolution = viewSize;
            }
            else {
                outputResolution = cv::Size{
                    std::stoi(FLAGS_output_resolution.substr(0, found)),
                    std::stoi(FLAGS_output_resolution.substr(found + 1, FLAGS_output_resolution.length()))
                };
            }
            outputTransform = OutputTransform(result->asRef<ImageResult>().resultImage.size(), outputResolution);
            outputResolution = outputTransform.computeResolution();

            view.renderResultData(result->asRef<ImageResult>(), outputResolution);
            auto key = 1;
            while (!(27 == key || 'q' == key || 'Q' == key)) {
                view.show();
                key = cv::waitKey(1);
                view.handleKey(key);
            }
            return 0;
        }

        // for stream of images
        while (keepRunning) {
            if (pipeline.isReadyToProcess()) {
                //--- Capturing frame
                auto startTime = std::chrono::steady_clock::now();
                curr_frame = cap->read();
                if (curr_frame.empty()) {
                    // Input stream is over
                    break;
                }

                frameNum = pipeline.submitData(ImageInputData(curr_frame),
                    std::make_shared<ImageMetaData>(curr_frame, startTime));
            }

            //--- Waiting for free input slot or output data available. Function will return immediately if any of them are available.
            pipeline.waitForData();

            //--- Checking for results and rendering data if it's ready
            //--- If you need just plain data without rendering - cast result's underlying pointer to ImageResult*
            //    and use your own processing instead of calling renderResultData().
            while ((result = pipeline.getResult()) && keepRunning) {
                if (framesProcessed == 0) {
                    if (found == std::string::npos) {
                        outputResolution = result->asRef<ImageResult>().resultImage.size();
                        cv::Size viewSize = view.getSize();
                        if (outputResolution.height > viewSize.height || outputResolution.width > viewSize.width)
                            outputResolution = viewSize;
                    }
                    else {
                        outputResolution = cv::Size{
                            std::stoi(FLAGS_output_resolution.substr(0, found)),
                            std::stoi(FLAGS_output_resolution.substr(found + 1, FLAGS_output_resolution.length()))
                        };
                    }

                    outputTransform = OutputTransform(result->asRef<ImageResult>().resultImage.size(), outputResolution);
                    outputResolution = outputTransform.computeResolution();

                    // Preparing video writer if needed
                    if (!FLAGS_o.empty() && !videoWriter.isOpened()) {
                        if (!videoWriter.open(FLAGS_o, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                            cap->fps(), outputResolution)) {
                            throw std::runtime_error("Can't open video writer");
                        }
                    }
                }

                cv::Mat outFrame = view.renderResultData(result->asRef<ImageResult>(), outputResolution);
                //--- Showing results and device information
                presenter.drawGraphs(outFrame);
                metrics.update(result->metaData->asRef<ImageMetaData>().timeStamp,
                    outFrame, { 10, 22 }, cv::FONT_HERSHEY_COMPLEX, 0.65);
                if (videoWriter.isOpened() && (FLAGS_limit == 0 || framesProcessed <= FLAGS_limit - 1)) {
                    videoWriter.write(outFrame);
                }
                if (!FLAGS_no_show) {
                    view.show(outFrame);

                    //--- Processing keyboard events
                    auto key = cv::waitKey(1);
                    if (27 == key || 'q' == key || 'Q' == key) { // Esc
                        keepRunning = false;
                    } else {
                        view.handleKey(key);
                        presenter.handleKey(key);
                    }
                }
                framesProcessed++;
            }
        } // while(keepRunning)

        //// ------------ Waiting for completion of data processing and rendering the rest of results ---------
        pipeline.waitForTotalCompletion();

        for (; framesProcessed <= frameNum; framesProcessed++)
        {
            result = pipeline.getResult();
            if (result != nullptr)
            {
                if (framesProcessed == 0) {
                    if (found == std::string::npos) {
                        outputResolution = result->asRef<ImageResult>().resultImage.size();
                        cv::Size viewSize = view.getSize();
                        if (outputResolution.height > viewSize.height || outputResolution.width > viewSize.width)
                            outputResolution = viewSize;
                    }
                    else {
                        outputResolution = cv::Size{
                            std::stoi(FLAGS_output_resolution.substr(0, found)),
                            std::stoi(FLAGS_output_resolution.substr(found + 1, FLAGS_output_resolution.length()))
                        };
                    }
                    outputTransform = OutputTransform(result->asRef<ImageResult>().resultImage.size(), outputResolution);
                    outputResolution = outputTransform.computeResolution();

                    // Preparing video writer if needed
                    if (!FLAGS_o.empty() && !videoWriter.isOpened()) {
                        if (!videoWriter.open(FLAGS_o, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                            cap->fps(), outputResolution)) {
                            throw std::runtime_error("Can't open video writer");
                        }
                    }
                }

                cv::Mat outFrame = view.renderResultData(result->asRef<ImageResult>(), outputResolution);
                //--- Showing results and device information
                presenter.drawGraphs(outFrame);
                metrics.update(result->metaData->asRef<ImageMetaData>().timeStamp,
                    outFrame, { 10, 22 }, cv::FONT_HERSHEY_COMPLEX, 0.65);
                if (videoWriter.isOpened() && (FLAGS_limit == 0 || framesProcessed <= FLAGS_limit - 1)) {
                    videoWriter.write(outFrame);
                }
                if (!FLAGS_no_show) {
                    view.show(outFrame);

                    //--- Updating output window
                    cv::waitKey(1);
                }
            }
        }

        //// --------------------------- Report metrics -------------------------------------------------------
        slog::info << slog::endl << "Metric reports:" << slog::endl;
        metrics.printTotal();

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

    slog::info << slog::endl << "The execution has completed successfully" << slog::endl;
    return 0;
}
