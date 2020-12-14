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
#include <samples/images_capture.h>
#include <samples/default_flags.hpp>
#include <samples/performance_metrics.hpp>
#include <unordered_map>
#include "samples/params_parser.h"

#include <pipelines/async_pipeline.h>
#include <pipelines/config_factory.h>
#include <pipelines/metadata.h>
#include <models/detection_model_yolo.h>
#include <models/detection_model_ssd.h>

void PrepareParamsParser(ParamsParser& parser) {
    parser.addParam("help", "h", "", "Print a usage message.", false, 0);
    parser.addParam("architecture_type", "at", "", "Required. Architecture type: ssd or yolo", true);
    parser.addParam("input", "i", "", "Required. Path to a video input (image, video, cameraID or directory with images)"
                    " or multiple comma-separarted inputs.", true, -1);
    parser.addParam("loop", "", "", "Required. Loop input", false);
    parser.addParam("model", "m", "", "Required. Path to an .xml file with a trained model.", true, 1);
    parser.addParam("device", "d", "CPU", "Optional. Specify the target device to infer on (the list of available devices is shown below)."
                    "Default value is CPU. Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. "
                    "The demo will look for a suitable plugin for a specified device.", false, 1, true);
    parser.addParam("labels", "", "", "Optional. Path to a file with labels mapping.", false, 1);
    parser.addParam("perf_count", "pc", "", "Optional. Enables per-layer performance report.", false, 0);
    parser.addParam("cldnn", "c", "", "Required for GPU custom kernels. "
                    "Absolute path to the .xml file with the kernel descriptions.", false, 1);
    parser.addParam("layers", "l", "", "Required for CPU custom layers. "
                    "Absolute path to a shared library with the kernel implementations.", false, 1);
    parser.addParam("thresh", "t", "0.5", "Optional. Probability threshold for detections.", false, 1);
    parser.addParam("raw_output", "r", "", "Optional. Inference results as raw values.", false, 0);
    parser.addParam("auto_resize", "", "", "Optional. Enables resizable input with support of ROI crop & auto resize.", false, 0);
    parser.addParam("nireq", "", "0", "Optional. Number of infer requests. If this option is omitted number of infer requests will be determined automatically.", false, 1);
    parser.addParam("nthreads", "", "0", "Optional. Number of threads.", false, 1);
    parser.addParam("nstreams", "", "", "Optional. Number of streams to use for inference on the CPU or/and GPU in "
                    "throughput mode (for HETERO and MULTI device cases use format "
                    "<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>)", false, 1, true);
    parser.addParam("no_show", "", "", "Optional. Do not show processed video.", false, 0);
    parser.addParam("utilization", "u", "", "Optional. List of monitors to show initially.", false);
    parser.addParam("iou_t", "", "0.4", "Optional. Filtering intersection over union threshold for overlapping boxes (YOLOv3 only).", false, 1);
    parser.addParam("yolo_af", "", "", "Optional. Use advanced postprocessing/filtering algorithm for YOLO.", false, 0);
}


// Input image is stored inside metadata, as we put it there during submission stage
cv::Mat renderDetectionData(const DetectionResult& result, bool showRawOutput) {
    if (!result.metaData) {
        throw std::invalid_argument("Renderer: metadata is null");
    }

    auto outputImg = result.metaData->asRef<ImageMetaData>().img;

    if (outputImg.empty()) {
        throw std::invalid_argument("Renderer: image provided in metadata is empty");
    }

    // Visualizing result data over source image
    if (showRawOutput) {
        slog::info << " Class ID  | Confidence | XMIN | YMIN | XMAX | YMAX " << slog::endl;
    }

    for (auto obj : result.objects) {
        if (showRawOutput) {
            slog::info << " "
                       << std::left << std::setw(9) << obj.label << " | "
                       << std::setw(10) << obj.confidence << " | "
                       << std::setw(4) << std::max(int(obj.x), 0) << " | "
                       << std::setw(4) << std::max(int(obj.y), 0) << " | "
                       << std::setw(4) << std::min(int(obj.width), outputImg.cols) << " | "
                       << std::setw(4) << std::min(int(obj.height), outputImg.rows)
                       << slog::endl;
        }

        std::ostringstream conf;
        conf << ":" << std::fixed << std::setprecision(3) << obj.confidence;

        cv::putText(outputImg, obj.label + conf.str(),
            cv::Point2f(obj.x, obj.y - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1,
            cv::Scalar(0, 0, 255));
        cv::rectangle(outputImg, obj, cv::Scalar(0, 0, 255));
    }

    return outputImg;
}


int main(int argc, char *argv[]) {
    try {
        PerformanceMetrics metrics;

        slog::info << "InferenceEngine: " << printable(*InferenceEngine::GetInferenceEngineVersion()) << slog::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        ParamsParser parser;
        PrepareParamsParser(parser);
        std::string parserErr = parser.parse(argc, argv, true);

        if (parser["help"]) {
            std::cout << std::endl;
            std::cout << "object_detection_demo [OPTION]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << parser.GetParamsHelp();
            showAvailableDevices();
            return false;
        }

        if(parserErr!="")
        {
            throw std::invalid_argument(parserErr);
        }

        //------------------------------- Preparing Input ------------------------------------------------------
        slog::info << "Reading input" << slog::endl;
        auto cap = openImagesCapture(parser.getStr("input"), parser["loop"]);
        cv::Mat curr_frame;

        //------------------------------ Running Detection routines ----------------------------------------------
        std::vector<std::string> labels;
        if (parser["labels"])
            labels = DetectionModel::loadLabels(parser.getStr("labels"));

        std::unique_ptr<ModelBase> model;
        if (parser.getStr("architecture_type") == "ssd") {
            model.reset(new ModelSSD(parser.getStr("model"), (float)parser.getDbl("thresh"), parser["auto_resize"], labels));
        }
        else if (parser.getStr("architecture_type") == "yolo") {
            model.reset(new ModelYolo3(parser.getStr("model"), (float)parser.getDbl("thresh"), parser["auto_resize"],
                                       parser["yolo_af"], (float)parser.getDbl("iou_t"), labels));
        }
        else {
            slog::err << "No model type or invalid model type (-at) provided: " + parser.getStr("architecture_type") << slog::endl;
            return -1;
        }

        InferenceEngine::Core core;
        AsyncPipeline pipeline(std::move(model),
            ConfigFactory::getUserConfig(parser.getStr("device"),
                                         parser.getStr("layers"),
                                         parser.getStr("cldnn"),
                                         parser["perf_count"],
                                         parser.getInt("nireq"),
                                         parser.getStr("nstreams"),
                                         parser.getInt("nthreads")),
                               core);
        Presenter presenter(parser.getStr("utilization"));

        bool keepRunning = true;
        int64_t frameNum = -1;
        std::unique_ptr<ResultBase> result;

        while (keepRunning) {
            if (pipeline.isReadyToProcess()) {
                //--- Capturing frame
                auto startTime = std::chrono::steady_clock::now();
                curr_frame = cap->read();
                if (curr_frame.empty()) {
                    if (frameNum == -1) {
                        throw std::logic_error("Can't read an image from the input");
                    }
                    else {
                        // Input stream is over
                        break;
                    }
                }

                frameNum = pipeline.submitData(ImageInputData(curr_frame),
                    std::make_shared<ImageMetaData>(curr_frame, startTime));
            }

            //--- Waiting for free input slot or output data available. Function will return immediately if any of them are available.
            pipeline.waitForData();

            //--- Checking for results and rendering data if it's ready
            //--- If you need just plain data without rendering - cast result's underlying pointer to DetectionResult*
            //    and use your own processing instead of calling renderDetectionData().
            while ((result = pipeline.getResult()) && keepRunning) {
                cv::Mat outFrame = renderDetectionData(result->asRef<DetectionResult>(), parser["raw_output"]);
                //--- Showing results and device information
                presenter.drawGraphs(outFrame);
                metrics.update(result->metaData->asRef<ImageMetaData>().timeStamp,
                    outFrame, { 10, 22 }, cv::FONT_HERSHEY_COMPLEX, 0.65);
                if (!parser["no_show"]) {

                    cv::imshow("Detection Results", outFrame);
                    //--- Processing keyboard events
                    int key = cv::waitKey(1);
                    if (27 == key || 'q' == key || 'Q' == key) {  // Esc
                        keepRunning = false;
                    }
                    else {
                        presenter.handleKey(key);
                    }
                }
            }
        }

        //// ------------ Waiting for completion of data processing and rendering the rest of results ---------
        pipeline.waitForTotalCompletion();
        while (result = pipeline.getResult()) {
            cv::Mat outFrame = renderDetectionData(result->asRef<DetectionResult>(), parser["raw_output"]);
            //--- Showing results and device information
            presenter.drawGraphs(outFrame);
            metrics.update(result->metaData->asRef<ImageMetaData>().timeStamp,
                outFrame, { 10, 22 }, cv::FONT_HERSHEY_COMPLEX, 0.65);
            if (!parser["no_show"]) {
                cv::imshow("Detection Results", outFrame);
                //--- Updating output window
                cv::waitKey(1);
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
