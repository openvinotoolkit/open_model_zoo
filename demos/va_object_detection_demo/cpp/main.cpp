/*
// Copyright (C) 2018-2021 Intel Corporation
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

#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <random>

#include <monitors/presenter.h>
#include <utils/ocv_common.hpp>
#include <utils/args_helper.hpp>
#include <utils/slog.hpp>
#include <utils/default_flags.hpp>
#include <utils/performance_metrics.hpp>
#include <unordered_map>
#include <gflags/gflags.h>

#include <gpu/gpu_context_api_va.hpp>
#include <ie_compound_blob.h>
#include "gst_vaapi_decoder.h"
#include "utils/config_factory.h"
#include "models/results.h"
#include <cldnn/cldnn_config.hpp>

DEFINE_INPUT_FLAGS
DEFINE_OUTPUT_FLAGS

static const char help_message[] = "Print a usage message.";
static const char model_message[] = "Required. Path to an .xml file with a trained model.";
static const char performance_counter_message[] = "Optional. Enables per-layer performance report.";
static const char custom_cldnn_message[] = "Required for GPU custom kernels. "
"Absolute path to the .xml file with the kernel descriptions.";
static const char thresh_output_message[] = "Optional. Probability threshold for detections.";
static const char raw_output_message[] = "Optional. Inference results as raw values.";
static const char no_show_message[] = "Optional. Don't show output.";
static const char utilization_monitors_message[] = "Optional. List of monitors to show initially.";

DEFINE_bool(h, false, help_message);
DEFINE_string(m, "", model_message);
DEFINE_bool(pc, false, performance_counter_message);
DEFINE_string(c, "", custom_cldnn_message);
DEFINE_bool(r, false, raw_output_message);
DEFINE_double(t, 0.5, thresh_output_message);
DEFINE_bool(no_show, false, no_show_message);
DEFINE_string(u, "", utilization_monitors_message);

struct CNNDesc
{
    std::string imgInputName;
    std::string infoInputName;
    std::string outputName;

    size_t netInputHeight;
    size_t netInputWidth;
    size_t maxProposalCount;
    size_t objectSize;
};


/**
* \brief This function shows a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "va_object_detection_demo [OPTION]" << std::endl;
    std::cout << "This demo runs only on GPU device." << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                        " << help_message << std::endl;
    std::cout << "    -i                        " << input_message << std::endl;
    std::cout << "    -m \"<path>\"               " << model_message << std::endl;
    std::cout << "    -o \"<path>\"               " << output_message << std::endl;
    std::cout << "    -limit \"<num>\"            " << limit_message << std::endl;
    std::cout << "      -c \"<absolute_path>\"    " << custom_cldnn_message << std::endl;
    std::cout << "    -pc                       " << performance_counter_message << std::endl;
    std::cout << "    -r                        " << raw_output_message << std::endl;
    std::cout << "    -t                        " << thresh_output_message << std::endl;
    std::cout << "    -loop                     " << loop_message << std::endl;
    std::cout << "    -no_show                  " << no_show_message << std::endl;
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

    if (!FLAGS_output_resolution.empty() && FLAGS_output_resolution.find("x") == std::string::npos) {
        throw std::logic_error("Correct format of -output_resolution parameter is \"width\"x\"height\".");
    }
    return true;
}

// Input image is stored inside metadata, as we put it there during submission stage
cv::Mat renderDetectionData(const std::vector<DetectedObject>& result, const cv::Mat& outputImg) {
    // Visualizing result data
    if (FLAGS_r) {
        slog::info << " Class ID  | Confidence | XMIN | YMIN | XMAX | YMAX " << slog::endl;
    }

    for (auto& obj : result) {
        if (FLAGS_r) {
            slog::info << " "
                << std::left << std::setw(9) << obj.label << " | "
                << std::setw(10) << obj.confidence << " | "
                << std::setw(4) << std::max(int(obj.x), 0) << " | "
                << std::setw(4) << std::max(int(obj.y), 0) << " | "
                << std::setw(4) << std::min(int(obj.x + obj.width), outputImg.cols) << " | "
                << std::setw(4) << std::min(int(obj.y + obj.height), outputImg.rows)
                << slog::endl;
        }

        std::ostringstream conf;
        conf << ":" << std::fixed << std::setprecision(1) << obj.confidence * 100 << '%';
        cv::Scalar color(0,0,192);
        cv::putText(outputImg, obj.label + conf.str(),
            cv::Point2f(obj.x, obj.y - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, { 230, 230, 230 }, 3);
        cv::putText(outputImg, obj.label + conf.str(),
            cv::Point2f(obj.x, obj.y - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, color);
        cv::rectangle(outputImg, obj, color, 2);
    }

    return outputImg;
}

InferenceEngine::ExecutableNetwork loadNetwork(InferenceEngine::Core& core, const std::string& modelFileName,
    InferenceEngine::gpu::VAContext::Ptr vaSharedContext,
    const std::string& clKernelsConfigPath, bool enablePerfCounters, CNNDesc& cnnDesc) {
    // --------------------------- Load inference engine ------------------------------------------------
    slog::info << "Loading Inference Engine" << slog::endl;

    slog::info << "Device info: " << slog::endl;
    slog::info << printable(core.GetVersions("GPU"));

    /** Load extensions for the plugin **/
    if (!clKernelsConfigPath.empty()) {
        // clDNN Extensions are loaded from an .xml description and OpenCL kernel files
        core.SetConfig({ {InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, clKernelsConfigPath} }, "GPU");
    }

    // --------------------------- Read IR Generated by ModelOptimizer (.xml and .bin files) ------------
    slog::info << "Loading network files" << slog::endl;
    /** Read network model **/
    InferenceEngine::CNNNetwork cnnNetwork = core.ReadNetwork(modelFileName);
    /** Set batch size to 1 **/
    slog::info << "Batch size is forced to 1." << slog::endl;
    auto shapes = cnnNetwork.getInputShapes();
    for (auto& shape : shapes)
        shape.second[0] = 1;
    cnnNetwork.reshape(shapes);


    // -------------------------- Reading all outputs names and customizing I/O blobs
        // --------------------------- Configure input & output -------------------------------------------------
    // --------------------------- Prepare input blobs ------------------------------------------------------
    slog::info << "Checking that the inputs are as the demo expects" << slog::endl;
    InferenceEngine::InputsDataMap inputInfo(cnnNetwork.getInputsInfo());

    for (const auto& inputInfoItem : inputInfo) {
        if (inputInfoItem.second->getTensorDesc().getDims().size() == 4) {  // 1st input contains images
            cnnDesc.imgInputName = inputInfoItem.first;

            inputInfoItem.second->setPrecision(InferenceEngine::Precision::U8);
            const InferenceEngine::TensorDesc& inputDesc = inputInfoItem.second->getTensorDesc();
            cnnDesc.netInputHeight = getTensorHeight(inputDesc);
            cnnDesc.netInputWidth = getTensorWidth(inputDesc);

            inputInfoItem.second->getPreProcess().setColorFormat(InferenceEngine::ColorFormat::NV12);

        }
        else if (inputInfoItem.second->getTensorDesc().getDims().size() == 2) {  // 2nd input contains image info
            cnnDesc.infoInputName = inputInfoItem.first;
            inputInfoItem.second->setPrecision(InferenceEngine::Precision::FP32);
        }
        else {
            throw std::logic_error("Unsupported " +
                std::to_string(inputInfoItem.second->getTensorDesc().getDims().size()) + "D "
                "input layer '" + inputInfoItem.first + "'. "
                "Only 4D input layers are supported");
        }
    }

    // --------------------------- Prepare output blobs -----------------------------------------------------
    slog::info << "Checking that the outputs are as the demo expects" << slog::endl;
    InferenceEngine::OutputsDataMap outputInfo(cnnNetwork.getOutputsInfo());
    InferenceEngine::DataPtr& output = outputInfo.begin()->second;
    cnnDesc.outputName = outputInfo.begin()->first;

    const InferenceEngine::SizeVector outputDims = output->getTensorDesc().getDims();

    if (outputDims.size() != 4) {
        throw std::logic_error("Incorrect output dimensions for SSD");
    }

    cnnDesc.maxProposalCount = outputDims[2];
    cnnDesc.objectSize = outputDims[3];
    if (cnnDesc.objectSize != 7) {
        throw std::logic_error("Output should have 7 as a last dimension");
    }

    output->setPrecision(InferenceEngine::Precision::FP32);
    output->setLayout(InferenceEngine::Layout::NCHW);

    // -------------------------- Configuring executable network and loading it
    slog::info << "Loading model to the device" << slog::endl;
        std::map<std::string, std::string> execNetworkConfig;
        if (enablePerfCounters) {
            execNetworkConfig.emplace(CONFIG_KEY(PERF_COUNT), InferenceEngine::PluginConfigParams::YES);
        }
        execNetworkConfig.emplace(CONFIG_KEY(GPU_THROUGHPUT_STREAMS), "1");
        execNetworkConfig.emplace(InferenceEngine::CLDNNConfigParams::KEY_CLDNN_NV12_TWO_INPUTS,
            InferenceEngine::PluginConfigParams::YES);

    return core.LoadNetwork(cnnNetwork, vaSharedContext, execNetworkConfig);
}

std::vector<DetectedObject> postprocess(const InferenceEngine::Blob::Ptr& outBlob, const CNNDesc& cnnDesc,
    int inputImgWidth, int inputImgHeight, float confidenceThreshold) {

    InferenceEngine::LockedMemory<const void> outputMapped = InferenceEngine::as<InferenceEngine::TBlob<float>>(outBlob)->rmap();
    const float *detections = outputMapped.as<float*>();

    std::vector<DetectedObject> result;

    for (size_t i = 0; i < cnnDesc.maxProposalCount; i++) {
        float image_id = detections[i * cnnDesc.objectSize + 0];
        if (image_id < 0) {
            break;
        }

        float confidence = detections[i * cnnDesc.objectSize + 2];

        /** Filtering out objects with confidence < confidence_threshold probability **/
        if (confidence > confidenceThreshold) {
            DetectedObject desc;

            desc.confidence = confidence;
            desc.labelID = static_cast<int>(detections[i * cnnDesc.objectSize + 1]);
            desc.label = std::string("Label#")+std::to_string(desc.labelID);
            desc.x = detections[i * cnnDesc.objectSize + 3] * inputImgWidth;
            desc.y = detections[i * cnnDesc.objectSize + 4] * inputImgHeight;
            desc.width = detections[i * cnnDesc.objectSize + 5] * inputImgWidth - desc.x;
            desc.height = detections[i * cnnDesc.objectSize + 6] * inputImgHeight - desc.y;

            result.push_back(desc);
        }
    }

    return result;
}



int main(int argc, char *argv[]) {
    try {
        PerformanceMetrics fullMetrics;
        PerformanceMetrics decodingMetrics;
        PerformanceMetrics preprocessingMetrics;
        PerformanceMetrics inferenceMetrics;
        PerformanceMetrics postprocessingMetrics;
        PerformanceMetrics renderingMetrics;
        InferenceEngine::Core core;

        slog::info << "InferenceEngine: " << printable(*InferenceEngine::GetInferenceEngineVersion()) << slog::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        //------------------------------- Preparing Input ------------------------------------------------------
        slog::info << "Reading input" << slog::endl;

        auto vaContext = std::make_shared<VaApiContext>(core);
        auto sharedContext = vaContext->sharedContext();

        GstVaApiDecoder decoder;
        VaApiImage::Ptr srcImage;

        decoder.open(FLAGS_i);
        decoder.play();

        //------------------------------ Running Detection routines ----------------------------------------------

        Presenter presenter(FLAGS_u);

        bool keepRunning = true;
        uint32_t framesProcessed = 0;

        cv::VideoWriter videoWriter;
        double videoFps = 30;

        CNNDesc cnnDesc;
        auto execNetwork = loadNetwork(core, FLAGS_m, sharedContext, FLAGS_c, FLAGS_pc, cnnDesc);
        auto inferRequest = execNetwork.CreateInferRequest();

        while (keepRunning) {
            auto startTime = std::chrono::steady_clock::now();

            if (!decoder.read(srcImage)) {
                // Input stream is over
                break;
            }
            decodingMetrics.update(startTime);

            // Resizing image to network's input size and putting it into blob
            auto preprocessingStartTime = std::chrono::steady_clock::now();
            auto resizedImg = srcImage->cloneToAnotherContext(vaContext)->
                resizeUsingPooledSurface(cnnDesc.netInputWidth, cnnDesc.netInputHeight, RESIZE_FILL,false);
            inferRequest.SetBlob(cnnDesc.imgInputName,
                InferenceEngine::gpu::make_shared_blob_nv12(cnnDesc.netInputHeight, cnnDesc.netInputWidth,
                sharedContext, resizedImg->va_surface_id));
            preprocessingMetrics.update(preprocessingStartTime);

            // Inferring and postprocessing the result
            auto inferenceStartTime = std::chrono::steady_clock::now();
            inferRequest.Infer();
            inferenceMetrics.update(inferenceStartTime);

            auto postprocessStartTime = std::chrono::steady_clock::now();
            auto result = postprocess(inferRequest.GetBlob(cnnDesc.outputName),cnnDesc,srcImage->width,srcImage->height,FLAGS_t);
            postprocessingMetrics.update(postprocessStartTime);

            videoFps = decoder.getFPS();

            // Preparing video writer if needed
            if (!FLAGS_o.empty() && !videoWriter.isOpened()) {
                if (!videoWriter.open(FLAGS_o, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                    videoFps, cv::Size(srcImage->width,srcImage->height))) {
                    throw std::runtime_error("Can't open video writer");
                }
            }

            //--- Checking for results and rendering data if it's ready
            //--- If you need just plain data without rendering - cast result's underlying pointer to DetectionResult*
            //    and use your own processing instead of calling renderDetectionData().
            auto renderingStartTime = std::chrono::steady_clock::now();
            cv::Mat outFrame = renderDetectionData(result, srcImage->copyToMat());
            renderingMetrics.update(renderingStartTime);

            //--- Showing results and device information
            presenter.drawGraphs(outFrame);

            fullMetrics.update(startTime,
                outFrame, { 10, 22 }, cv::FONT_HERSHEY_COMPLEX, 0.65);
            if (videoWriter.isOpened() && (FLAGS_limit == 0 || framesProcessed <= FLAGS_limit - 1)) {
                videoWriter.write(outFrame);
            }
            framesProcessed++;

            if (!FLAGS_no_show) {
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

        //// --------------------------- Report metrics -------------------------------------------------------
        slog::info << slog::endl << "Metric reports:" << slog::endl;
        fullMetrics.printTotal();
        slog::info << slog::endl << "Avg time:\n";
        slog::info << "  * Decoding : \t\t" << std::fixed << std::setprecision(2) <<
            decodingMetrics.getTotal().latency
            << " ms\n";
         slog::info << "  * Preprocessing :\t" << std::fixed << std::setprecision(2) <<
            preprocessingMetrics.getTotal().latency << " ms\n";
        slog::info << "  * Inference :\t\t" << std::fixed << std::setprecision(2) <<
            inferenceMetrics.getTotal().latency << " ms\n";
        slog::info << "  * Postprocessing :\t" << std::fixed << std::setprecision(2) <<
            postprocessingMetrics.getTotal().latency << " ms\n";
        slog::info << "  * Rendering :\t\t" << std::fixed << std::setprecision(2) <<
            renderingMetrics.getTotal().latency << " ms" << slog::endl;

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
