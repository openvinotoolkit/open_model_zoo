/*
// Copyright (C) 2021-2024 Intel Corporation
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

#include <stddef.h>
#include <stdint.h>

#include <chrono>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gflags/gflags.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

#include <models/image_model.h>
#include <models/input_data.h>
#include <models/jpeg_restoration_model.h>
#include <models/model_base.h>
#include <models/results.h>
#include <models/style_transfer_model.h>
#include <models/super_resolution_model.h>
#include <monitors/presenter.h>
#include <pipelines/async_pipeline.h>
#include <pipelines/metadata.h>
#include <utils/common.hpp>
#include <utils/config_factory.h>
#include <utils/default_flags.hpp>
#include <utils/images_capture.h>
#include <utils/ocv_common.hpp>
#include <utils/performance_metrics.hpp>
#include <utils/slog.hpp>

#include "visualizer.hpp"

DEFINE_INPUT_FLAGS
DEFINE_OUTPUT_FLAGS

static const char help_message[] = "Print a usage message.";
static const char at_message[] = "Required. Type of the model, either 'sr' for Super Resolution task, "
                                 "'sr_channel_joint' for Super Resolution model that accepts and returns 1 channel image, "
                                 "'jr' for JPEGRestoration, 'style' for Style Transfer task.";
static const char model_message[] = "Required. Path to an .xml file with a trained model.";
static const char layout_message[] = "Optional. Specify inputs layouts."
                                     " Ex. NCHW or input0:NCHW,input1:NC in case of more than one input.";
static const char target_device_message[] =
    "Optional. Specify the target device to infer on (the list of available devices is shown below). "
    "Default value is CPU. Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. "
    "The demo will look for a suitable plugin for a specified device.";
static const char nireq_message[] = "Optional. Number of infer requests. If this option is omitted, number of infer "
                                    "requests is determined automatically.";
static const char num_threads_message[] = "Optional. Number of threads.";
static const char num_streams_message[] = "Optional. Number of streams to use for inference on the CPU or/and GPU in "
                                          "throughput mode (for HETERO and MULTI device cases use format "
                                          "<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>)";
static const char no_show_processed_video[] = "Optional. Do not show processed video. If disabled and --output_resolution isn't set, "
    "the resulting image is resized to a default view size: 1000x600 but keeping the aspect ratio";
static const char utilization_monitors_message[] = "Optional. List of monitors to show initially.";
static const char output_resolution_message[] =
    "Optional. Specify the maximum output window resolution "
    "in (width x height) format. Example: 1280x720. Input frame size used by default.";
static const char jc_message[] = "Optional. Flag of using compression for jpeg images. "
                                 "Default value if false. Only for jr architecture type.";
static const char reverse_input_channels_message[] = "Optional. Switch the input channels order from BGR to RGB.";
static const char mean_values_message[] =
    "Optional. Normalize input by subtracting the mean values per channel. Example: \"255.0 255.0 255.0\"";
static const char scale_values_message[] = "Optional. Divide input by scale values per channel. Division is applied "
                                           "after mean values subtraction. Example: \"255.0 255.0 255.0\"";

DEFINE_bool(h, false, help_message);
DEFINE_string(at, "", at_message);
DEFINE_string(m, "", model_message);
DEFINE_string(layout, "", layout_message);
DEFINE_string(d, "CPU", target_device_message);
DEFINE_uint32(nireq, 0, nireq_message);
DEFINE_uint32(nthreads, 0, num_threads_message);
DEFINE_string(nstreams, "", num_streams_message);
DEFINE_bool(no_show, false, no_show_processed_video);
DEFINE_string(u, "", utilization_monitors_message);
DEFINE_string(output_resolution, "", output_resolution_message);
DEFINE_bool(jc, false, jc_message);
DEFINE_bool(reverse_input_channels, false, reverse_input_channels_message);
DEFINE_string(mean_values, "", mean_values_message);
DEFINE_string(scale_values, "", scale_values_message);

/**
 * \brief This function shows a help message
 */
static void showUsage() {
    std::cout << std::endl;
    std::cout << "image_processing_demo [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                        " << help_message << std::endl;
    std::cout << "    -at \"<type>\"              " << at_message << std::endl;
    std::cout << "    -i \"<path>\"               " << input_message << std::endl;
    std::cout << "    -m \"<path>\"               " << model_message << std::endl;
    std::cout << "    -layout \"<string>\"        " << layout_message << std::endl;
    std::cout << "    -o \"<path>\"               " << output_message << std::endl;
    std::cout << "    -limit \"<num>\"            " << limit_message << std::endl;
    std::cout << "    -d \"<device>\"             " << target_device_message << std::endl;
    std::cout << "    -nireq \"<integer>\"        " << nireq_message << std::endl;
    std::cout << "    -nthreads \"<integer>\"     " << num_threads_message << std::endl;
    std::cout << "    -nstreams                 " << num_streams_message << std::endl;
    std::cout << "    -loop                     " << loop_message << std::endl;
    std::cout << "    -no_show                  " << no_show_processed_video << std::endl;
    std::cout << "    -output_resolution        " << output_resolution_message << std::endl;
    std::cout << "    -u                        " << utilization_monitors_message << std::endl;
    std::cout << "    -jc                       " << jc_message << std::endl;
    std::cout << "    -reverse_input_channels   " << reverse_input_channels_message << std::endl;
    std::cout << "    -mean_values              " << mean_values_message << std::endl;
    std::cout << "    -scale_values             " << scale_values_message << std::endl;
}

bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }
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

std::unique_ptr<ImageModel> getModel(const cv::Size& frameSize, const std::string& type, bool doCompression = false) {
    if (type == "sr") {
        return std::unique_ptr<ImageModel>(new SuperResolutionModel(FLAGS_m, frameSize, FLAGS_layout));
    }
    if (type == "sr_channel_joint") {
        return std::unique_ptr<ImageModel>(new SuperResolutionChannelJoint(FLAGS_m, frameSize, FLAGS_layout));
    }
    if (type == "jr") {
        return std::unique_ptr<ImageModel>(new JPEGRestorationModel(FLAGS_m, frameSize, doCompression, FLAGS_layout));
    }
    if (type == "style") {
        return std::unique_ptr<ImageModel>(new StyleTransferModel(FLAGS_m, FLAGS_layout));
    }
    throw std::invalid_argument("No model type or invalid model type (-at) provided: " + FLAGS_at);
}

int main(int argc, char* argv[]) {
    try {
        PerformanceMetrics metrics, renderMetrics;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        //------------------------------- Preparing Input ------------------------------------------------------
        auto cap = openImagesCapture(FLAGS_i, FLAGS_loop, FLAGS_nireq == 1 ? read_type::efficient : read_type::safe);
        cv::Mat curr_frame;

        auto startTime = std::chrono::steady_clock::now();
        curr_frame = cap->read();

        //------------------------------ Running ImageProcessing routines ----------------------------------------------
        slog::info << ov::get_openvino_version() << slog::endl;
        ov::Core core;

        std::unique_ptr<ImageModel> model = getModel(cv::Size(curr_frame.cols, curr_frame.rows), FLAGS_at, FLAGS_jc);
        model->setInputsPreprocessing(FLAGS_reverse_input_channels, FLAGS_mean_values, FLAGS_scale_values);
        AsyncPipeline pipeline(std::move(model),
                               ConfigFactory::getUserConfig(FLAGS_d, FLAGS_nireq, FLAGS_nstreams, FLAGS_nthreads),
                               core);
        Presenter presenter(FLAGS_u);

        int64_t frameNum =
            pipeline.submitData(ImageInputData(curr_frame), std::make_shared<ImageMetaData>(curr_frame, startTime));

        bool keepRunning = true;
        std::unique_ptr<ResultBase> result;
        uint32_t framesProcessed = 0;
        LazyVideoWriter videoWriter{FLAGS_o, cap->fps(), FLAGS_limit};

        cv::Size outputResolution;
        OutputTransform outputTransform = OutputTransform();
        size_t found = FLAGS_output_resolution.find("x");
        Visualizer view(FLAGS_at);

        // interactive mode for single image
        if (cap->getType() == "IMAGE" && !FLAGS_loop && !FLAGS_no_show) {
            pipeline.waitForTotalCompletion();
            result = pipeline.getResult();
            metrics.update(result->metaData->asRef<ImageMetaData>().timeStamp);
            auto renderingStart = std::chrono::steady_clock::now();
            if (found == std::string::npos) {
                outputResolution = result->asRef<ImageResult>().resultImage.size();
                cv::Size viewSize = view.getSize();
                if (!FLAGS_no_show && (outputResolution.height > viewSize.height || outputResolution.width > viewSize.width))
                    outputResolution = viewSize;
            } else {
                outputResolution =
                    cv::Size{std::stoi(FLAGS_output_resolution.substr(0, found)),
                             std::stoi(FLAGS_output_resolution.substr(found + 1, FLAGS_output_resolution.length()))};
            }
            outputTransform = OutputTransform(result->asRef<ImageResult>().resultImage.size(), outputResolution);
            outputResolution = outputTransform.computeResolution();

            view.renderResultData(result->asRef<ImageResult>(), outputResolution);
            renderMetrics.update(renderingStart);
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

            //--- Waiting for free input slot or output data available. Function will return immediately if any of them
            // are available.
            pipeline.waitForData();

            //--- Checking for results and rendering data if it's ready
            //--- If you need just plain data without rendering - cast result's underlying pointer to ImageResult*
            //    and use your own processing instead of calling renderResultData().
            while ((result = pipeline.getResult()) && keepRunning) {
                auto renderingStart = std::chrono::steady_clock::now();
                if (framesProcessed == 0) {
                    if (found == std::string::npos) {
                        outputResolution = result->asRef<ImageResult>().resultImage.size();
                        cv::Size viewSize = view.getSize();
                        if (!FLAGS_no_show && (outputResolution.height > viewSize.height || outputResolution.width > viewSize.width))
                            outputResolution = viewSize;
                    } else {
                        outputResolution = cv::Size{
                            std::stoi(FLAGS_output_resolution.substr(0, found)),
                            std::stoi(FLAGS_output_resolution.substr(found + 1, FLAGS_output_resolution.length()))};
                    }

                    outputTransform =
                        OutputTransform(result->asRef<ImageResult>().resultImage.size(), outputResolution);
                    outputResolution = outputTransform.computeResolution();
                }

                cv::Mat outFrame = view.renderResultData(result->asRef<ImageResult>(), outputResolution);
                //--- Showing results and device information
                presenter.drawGraphs(outFrame);
                renderMetrics.update(renderingStart);
                metrics.update(result->metaData->asRef<ImageMetaData>().timeStamp,
                               outFrame,
                               {10, 22},
                               cv::FONT_HERSHEY_COMPLEX,
                               0.65);
                videoWriter.write(outFrame);
                if (!FLAGS_no_show) {
                    view.show(outFrame);

                    //--- Processing keyboard events
                    auto key = cv::waitKey(1);
                    if (27 == key || 'q' == key || 'Q' == key) {  // Esc
                        keepRunning = false;
                    } else {
                        view.handleKey(key);
                        presenter.handleKey(key);
                    }
                }
                framesProcessed++;
            }
        }  // while(keepRunning)

        // ------------ Waiting for completion of data processing and rendering the rest of results ---------
        pipeline.waitForTotalCompletion();

        for (; framesProcessed <= frameNum; framesProcessed++) {
            result = pipeline.getResult();
            if (result != nullptr) {
                if (framesProcessed == 0) {
                    if (found == std::string::npos) {
                        outputResolution = result->asRef<ImageResult>().resultImage.size();
                        cv::Size viewSize = view.getSize();
                        if (!FLAGS_no_show && (outputResolution.height > viewSize.height || outputResolution.width > viewSize.width))
                            outputResolution = viewSize;
                    } else {
                        outputResolution = cv::Size{
                            std::stoi(FLAGS_output_resolution.substr(0, found)),
                            std::stoi(FLAGS_output_resolution.substr(found + 1, FLAGS_output_resolution.length()))};
                    }
                    outputTransform =
                        OutputTransform(result->asRef<ImageResult>().resultImage.size(), outputResolution);
                    outputResolution = outputTransform.computeResolution();
                }

                cv::Mat outFrame = view.renderResultData(result->asRef<ImageResult>(), outputResolution);
                //--- Showing results and device information
                presenter.drawGraphs(outFrame);
                metrics.update(result->metaData->asRef<ImageMetaData>().timeStamp,
                               outFrame,
                               {10, 22},
                               cv::FONT_HERSHEY_COMPLEX,
                               0.65);
                videoWriter.write(outFrame);
                if (!FLAGS_no_show) {
                    view.show(outFrame);

                    //--- Updating output window
                    cv::waitKey(1);
                }
            }
        }

        slog::info << "Metrics report:" << slog::endl;
        metrics.logTotal();
        logLatencyPerStage(cap->getMetrics().getTotal().latency,
                           pipeline.getPreprocessMetrics().getTotal().latency,
                           pipeline.getInferenceMetircs().getTotal().latency,
                           pipeline.getPostprocessMetrics().getTotal().latency,
                           renderMetrics.getTotal().latency);
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
