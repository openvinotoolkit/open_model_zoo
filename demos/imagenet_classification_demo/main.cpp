// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <queue>
#include <memory>
#include <string>
#include <map>
#include <chrono>
#include <condition_variable>
#include <mutex>

#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>
#include <samples/classification_results.h>
#include <samples/ocv_common.hpp>
#include <samples/common.hpp>

#include <vpu/vpu_plugin_config.hpp>
#include <cldnn/cldnn_config.hpp>

#include <opencv2/core.hpp>

#include "imagenet_classification_demo.hpp"
#include "grid_mat.hpp"
#include "infer_request_callback.hpp"

ConsoleErrorListener error_listener;

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------------Parsing and validation of input args----------------------------------
    slog::info << "Parsing input parameters" << slog::endl;

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

void resizeImage(cv::Mat& image, int modelInputResolution) {
    double scale = static_cast<double>(modelInputResolution) / std::min(image.cols, image.rows);
    cv::resize(image, image, cv::Size(), scale, scale);

    cv::Rect imgROI;
    if (image.cols >= image.rows) {
        int fromWidth = image.cols/2 - modelInputResolution/2;
        imgROI = cv::Rect(fromWidth,
                          0,
                          std::min(modelInputResolution, image.cols - fromWidth),
                          modelInputResolution);
    } else {
        int fromHeight = image.rows/2 - modelInputResolution/2;
        imgROI = cv::Rect(0,
                          fromHeight,
                          modelInputResolution,
                          std::min(modelInputResolution, image.rows - fromHeight));
    }
    image(imgROI).copyTo(image);
}

int main(int argc, char *argv[]) {
    try {
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }
            
        std::vector<std::string> imageNames;
        parseInputFilesArguments(imageNames);
        unsigned inputImagesCount = imageNames.size();
        if (inputImagesCount == 0) throw std::runtime_error("No images provided");

        // -------------------------------------------Read network--------------------------------------------
        InferenceEngine::Core ie;
        if (FLAGS_p_msg) {
            ie.SetLogCallback(error_listener);
        }
        InferenceEngine::CNNNetReader netReader;
        netReader.ReadNetwork(FLAGS_m);
        netReader.ReadWeights(fileNameNoExt(FLAGS_m) + ".bin");
        InferenceEngine::CNNNetwork network(netReader.getNetwork());

        // ----------------------------------------Init inputBlobName-----------------------------------------
        InferenceEngine::InputsDataMap inputInfo(network.getInputsInfo());
        if (inputInfo.size() != 1) throw std::logic_error("Sample supports topologies with 1 input only");
        std::string inputBlobName = inputInfo.begin()->first;

        // ------------------------------------------Reshape network------------------------------------------
        auto input_shapes = network.getInputShapes();
        std::string input_name;
        InferenceEngine::SizeVector input_shape;
        std::tie(input_name, input_shape) = *input_shapes.begin();
        input_shape[0] = FLAGS_b;
        input_shape[2] = input_shapes[input_name][2];
        input_shape[3] = input_shapes[input_name][3];
        input_shapes[input_name] = input_shape;
        std::cout << "Resizing network to the image size = [" 
                  << input_shapes[input_name][2] << "x" << input_shapes[input_name][3]
                  << "] " << "with batch = " << FLAGS_b << std::endl;
        network.reshape(input_shapes);

        // -----------------------------------------Configure layers------------------------------------------
        std::map<std::string, std::vector<unsigned long>> inputBlobsDimsInfo;
        std::map<std::string, std::vector<unsigned long>> outputBlobsDimsInfo;
        for (auto inputBlobsIt = inputInfo.begin(); inputBlobsIt != inputInfo.end(); ++inputBlobsIt) {
            auto layerName = inputBlobsIt->first;
            auto layerData = inputBlobsIt->second;
            auto layerDataDims = layerData->getTensorDesc().getDims();

            std::vector<unsigned long> layerDims(layerDataDims.data(), layerDataDims.data() + layerDataDims.size());
            inputBlobsDimsInfo[layerName] = layerDims;

            if (layerDataDims.size() == 4) {
                layerData->setLayout(InferenceEngine::Layout::NCHW);
                layerData->setPrecision(InferenceEngine::Precision::U8);
            } else if (layerDataDims.size() == 2) {
                layerData->setLayout(InferenceEngine::Layout::NC);
                layerData->setPrecision(InferenceEngine::Precision::FP32);
            } else {
                throw std::runtime_error("Unknown type of input layer layout. "
                                         "Expected either 4 or 2 dimensional inputs");
            }
        }

        auto outputInfo = network.getOutputsInfo();
        for (auto outputBlobsIt = outputInfo.begin(); outputBlobsIt != outputInfo.end(); ++outputBlobsIt) {
            auto layerName = outputBlobsIt->first;
            auto layerData = outputBlobsIt->second;
            auto layerDataDims = layerData->getTensorDesc().getDims();

            std::vector<unsigned long> layerDims(layerDataDims.data(), layerDataDims.data() + layerDataDims.size());
            outputBlobsDimsInfo[layerName] = layerDims;
            layerData->setPrecision(InferenceEngine::Precision::FP32);
        }
        // ---------------------------------------------------------------------------------------------------

        // ----------------------------------Set device and device settings-----------------------------------
        std::vector<std::string> deviceNames = parseDevices(FLAGS_d);
        std::set<std::string> devices(deviceNames.begin(), deviceNames.end());
        std::map<std::string, uint32_t> device_nstreams = parseValuePerDevice(devices, FLAGS_nstreams);
        for (auto& device : devices) {
            if (device == "CPU") {  // CPU supports a few special performance-oriented keys
                // limit threading for CPU portion of inference
                if (FLAGS_nthreads != 0)
                    ie.SetConfig({{ CONFIG_KEY(CPU_THREADS_NUM), std::to_string(FLAGS_nthreads) }}, device);

                if ((FLAGS_d.find("MULTI") != std::string::npos) &&
                    (FLAGS_d.find("GPU") != std::string::npos)) {
                    ie.SetConfig({{ CONFIG_KEY(CPU_BIND_THREAD), CONFIG_VALUE(NO) }}, device);
                } else {
                    // pin threads for CPU portion of inference
                    ie.SetConfig({{ CONFIG_KEY(CPU_BIND_THREAD), "YES" }}, device);
                }

                // for CPU execution, more throughput-oriented execution via streams
                ie.SetConfig({{ CONFIG_KEY(CPU_THROUGHPUT_STREAMS),
                                (device_nstreams.count(device) > 0 ? std::to_string(device_nstreams.at(device)) :
                                                                                 "CPU_THROUGHPUT_AUTO") }}, device);
                device_nstreams[device] = std::stoi(
                    ie.GetConfig(device, CONFIG_KEY(CPU_THROUGHPUT_STREAMS)).as<std::string>());
            } else if (device == ("GPU")) {
                ie.SetConfig({{ CONFIG_KEY(GPU_THROUGHPUT_STREAMS),
                                (device_nstreams.count(device) > 0 ? std::to_string(device_nstreams.at(device)) :
                                                                         "GPU_THROUGHPUT_AUTO") }}, device);
                device_nstreams[device] = std::stoi(
                    ie.GetConfig(device, CONFIG_KEY(GPU_THROUGHPUT_STREAMS)).as<std::string>());

                if ((FLAGS_d.find("MULTI") != std::string::npos) &&
                    (FLAGS_d.find("CPU") != std::string::npos)) {
                    // multi-device execution with the CPU + GPU performs best with GPU throttling hint,
                    // which releases another CPU thread (that is otherwise used by the GPU driver for active polling)
                    ie.SetConfig({{ CLDNN_CONFIG_KEY(PLUGIN_THROTTLE), "1" }}, "GPU");
                }
            } else if (device == "MYRIAD") {
                ie.SetConfig({{ CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_NONE) },
                              { VPU_CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_WARNING) }}, device);
            }
        }

        // --------------------------------------Load network to device---------------------------------------
        InferenceEngine::ExecutableNetwork executableNetwork;
        executableNetwork = ie.LoadNetwork(network, FLAGS_d);

        // ----------------------------Try to set optimal number of infer requests----------------------------
        if (FLAGS_nireq == 0) {
            std::string key = METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS);
            try {
                FLAGS_nireq = 2 * executableNetwork.GetMetric(key).as<unsigned int>();
            } catch (const InferenceEngine::details::InferenceEngineException& ex) {
                THROW_IE_EXCEPTION
                        << "Every device used with the imagenet_classification_demo should "
                        << "support OPTIMAL_NUMBER_OF_INFER_REQUESTS ExecutableNetwork metric. "
                        << "Failed to query the metric for the " << FLAGS_d << " with error:" << ex.what();
            }
        }

        // ---------------------------------------Create infer request----------------------------------------
        std::vector<InferenceEngine::InferRequest> inferRequests;
        int infReqNum = FLAGS_nireq;
        for (int infReqID = 0; infReqID < infReqNum; ++infReqID) {
            inferRequests.push_back(executableNetwork.CreateInferRequest());
        }
        std::list<cv::Mat> showMats;
        int modelInputResolution = input_shapes[input_name][2];
        
        // ----------------------------------------Create output info-----------------------------------------
        int width;
        int height;
        std::vector<std::string> gmRowsCols = split(FLAGS_res, 'x');        
        if (gmRowsCols.size() != 2) {
            throw std::runtime_error("The value of GridMat resolution flag is not valid.");
        } else {
            width = std::stoi(gmRowsCols[0]);
            height = std::stoi(gmRowsCols[1]);
        }
        
        GridMat gridMat = GridMat(inputImagesCount, cv::Size(width, height));

        unsigned spfResultsMaxCount = 1000;
        std::queue<double> spfQueue;
        double spf;
        double spfSum = 0;
        double avgFPS = 0;
        unsigned framesNum = 0;

        long long startTickCount = cv::getTickCount();
        auto startTime = std::chrono::system_clock::now();
        auto currentTime = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsedSeconds = currentTime - startTime;
        
        std::condition_variable condVar;
        std::mutex mutex;
        std::queue<std::pair<InferenceEngine::InferRequest&, std::vector<cv::Mat>>> emptyInferRequests;
        std::queue<std::pair<InferenceEngine::InferRequest&, std::vector<cv::Mat>>> completedInferRequests;
        for (std::size_t i = 0; i < inferRequests.size(); i++) {
            emptyInferRequests.push({inferRequests[i], std::vector<cv::Mat>()});
        }
        std::size_t nextImageIndex = 0;

        bool isTestMode = true;
        int delay = 1;
        char key = 0;
        do {
            if (irCallbackException) std::rethrow_exception(irCallbackException);

            if (isTestMode && elapsedSeconds.count() >= 10) {
                isTestMode = false;

                gridMat = GridMat(inputImagesCount, cv::Size(width, height), cv::Size(16, 9), avgFPS);
                cv::Size gridMatSize = gridMat.getSize();
                delay = ((FLAGS_delay == -1)
                         ? static_cast<int>(avgFPS / (gridMatSize.width * (gridMatSize.height / FLAGS_b)) * 1000)
                         : FLAGS_delay);
                
                spfSum = 0;
                avgFPS = 0;
                framesNum = 0;
                std::queue<double>().swap(spfQueue);
                startTickCount = cv::getTickCount();
            }

            std::unique_lock<std::mutex> lock(mutex);
        
            if (!completedInferRequests.empty()) {
                auto completedIR = completedInferRequests.front();
                completedInferRequests.pop();
                mutex.unlock();
                for (unsigned j = 0; j < FLAGS_b; j++) {
                    showMats.push_back(completedIR.second[j]);
                }
                framesNum += FLAGS_b;
                gridMat.listUpdate(showMats);
                spf = ((cv::getTickCount() - startTickCount) / cv::getTickFrequency()) / framesNum;
                if (spfQueue.size() >= spfResultsMaxCount) {
                    spfSum -= spfQueue.front();
                    spfQueue.pop();
                }
                spfQueue.push(spf);
                spfSum += spf;
                avgFPS = spfQueue.size() / spfSum;
                gridMat.textUpdate(avgFPS, isTestMode);
                if (!FLAGS_no_show && FLAGS_delay != -1) {
                    cv::imshow("main", gridMat.getMat());
                    key = static_cast<char>(cv::waitKey(delay));
                }
                emptyInferRequests.push({completedIR.first, std::vector<cv::Mat>()});
            }
            else if (!emptyInferRequests.empty()) {
                mutex.unlock();
                const cv::Mat& readImg = cv::imread(imageNames[nextImageIndex]);
                if (readImg.data == nullptr) {
                    std::cerr << "Could not read image " << imageNames[nextImageIndex] << '\n';
                } else {
                    cv::Mat tmpImg = readImg;
                    resizeImage(tmpImg, modelInputResolution);
                    emptyInferRequests.front().second.push_back(tmpImg);
                }
                nextImageIndex++;
                if (nextImageIndex == inputImagesCount) {
                    nextImageIndex = 0;
                }
                if (emptyInferRequests.front().second.size() == FLAGS_b) {
                    auto ir = emptyInferRequests.front();

                    ir.first.SetCompletionCallback(
                        InferRequestCallback(
                            ir.first,
                            ir.second,
                            completedInferRequests,
                            mutex,
                            condVar
                        )
                    );
                    
                    auto inputBlob = ir.first.GetBlob(inputBlobName);
                    for (unsigned i = 0; i < FLAGS_b; i++) {
                        matU8ToBlob<uint8_t>(ir.second[i], inputBlob, i);
                    }
                    ir.first.StartAsync();
                    emptyInferRequests.pop();
                }
            }
            
            mutex.try_lock();
            while (emptyInferRequests.empty() && completedInferRequests.empty()) {   
                condVar.wait(lock);
            }
            mutex.unlock();

            currentTime = std::chrono::system_clock::now();
            elapsedSeconds = currentTime - startTime;
        } while (27 != key && (FLAGS_time == -1 || elapsedSeconds.count() < FLAGS_time));

        std::cout << "Overall FPS: " << avgFPS << std::endl;

        if (!FLAGS_no_show) {
            cv::destroyWindow("main");
        }
        // ---------------------------------------------------------------------------------------------------
        
        // ------------------------------------Wait for all infer requests------------------------------------
        for (InferenceEngine::InferRequest& inferRequest : inferRequests)
            inferRequest.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
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
