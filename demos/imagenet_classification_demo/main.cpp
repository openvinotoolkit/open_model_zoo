// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <queue>
#include <memory>
#include <string>
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

#include "imagenet_classification_demo.hpp"
#include "grid_mat.hpp"
#include "infer_request_callback.hpp"

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

        // -----------------------------------------Read input images-----------------------------------------
        std::vector<std::string> imageNames;
        std::vector<cv::Mat> inputImages;
        parseInputFilesArguments(imageNames);
        unsigned inputImagesCount = imageNames.size();
        if (inputImagesCount == 0) throw std::runtime_error("No images provided");
        std::sort(imageNames.begin(), imageNames.end());
        for (size_t i = 0; i < inputImagesCount; i++) {
            std::string &name = imageNames[i];
            const cv::Mat& tmpImage = cv::imread(name);
            if (tmpImage.data == nullptr) {
                std::cerr << "Could not read image " << name << '\n';
                imageNames.erase(imageNames.begin() + i);
                inputImagesCount--;
                i--;
            }
            else {
                inputImages.push_back(tmpImage);
                name = name.substr(name.rfind('/') + 1);
            }
        }
        // ---------------------------------------------------------------------------------------------------

        // ----------------------------------------Read image classes-----------------------------------------
        std::vector<std::pair<std::string, std::string>> classIndexesMap;
        std::vector<std::string> classIndexes;
        std::string imageClassMappingFileName;
        const size_t lastSlashIdx = FLAGS_i.rfind('/');
        if (lastSlashIdx != std::string::npos) {
            imageClassMappingFileName = FLAGS_i + FLAGS_i.substr(lastSlashIdx) + ".txt";
        }
        else {
            throw std::runtime_error("No file provided for image->class mapping");
        }
        std::ifstream inputClassesFile(imageClassMappingFileName);
        while (true) {
            std::string imageName;
            std::string classIndex;
            inputClassesFile >> imageName >> classIndex;
            if (inputClassesFile.eof()) break;
            classIndexesMap.push_back({imageName, classIndex});
        }
        for (size_t i = 0; i < inputImagesCount; i++) {
            for (size_t j = 0; j < classIndexesMap.size(); j++) {
                if (imageNames[i] == classIndexesMap[j].first) {
                    classIndexes.push_back(classIndexesMap[j].second);
                    break;
                }
            }
            if (i+1 != classIndexes.size()) {
                throw std::runtime_error("No class specified for image " + imageNames[i]);
            }
        }
        classIndexesMap.clear();
        // ---------------------------------------------------------------------------------------------------

        // --------------------------------------------Read labels--------------------------------------------
        std::vector<std::string> labels;
        std::ifstream inputLabelsFile("imagenet_labels.txt");
        std::string labelsLine;
        while (std::getline(inputLabelsFile, labelsLine)) {
            labels.push_back(labelsLine.substr(0, labelsLine.find(',')));
        }
        // ---------------------------------------------------------------------------------------------------

        // -------------------------------------------Read network--------------------------------------------
        Core ie;
        CNNNetwork network = ie.ReadNetwork(FLAGS_m);
        // ---------------------------------------------------------------------------------------------------

        // ------------------------------------------Reshape network------------------------------------------
        auto input_shapes = network.getInputShapes();
        std::string input_name;
        SizeVector input_shape;
        std::tie(input_name, input_shape) = *input_shapes.begin();
        int modelInputResolution = input_shapes[input_name][2];
        input_shape[0] = FLAGS_b;
        input_shape[2] = input_shapes[input_name][2];
        input_shape[3] = input_shapes[input_name][3];
        input_shapes[input_name] = input_shape;
        std::cout << "Resizing network to the image size = [" 
                  << input_shapes[input_name][2] << "x" << input_shapes[input_name][3]
                  << "] " << "with batch = " << FLAGS_b << std::endl;
        network.reshape(input_shapes);
        // ---------------------------------------------------------------------------------------------------

        // ----------------------------------------Init inputBlobName-----------------------------------------
        InputsDataMap inputInfo(network.getInputsInfo());
        std::string inputBlobName = inputInfo.begin()->first;
        // ---------------------------------------------------------------------------------------------------

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
                layerData->setLayout(Layout::NCHW);
                layerData->setPrecision(Precision::U8);
            } else if (layerDataDims.size() == 2) {
                layerData->setLayout(Layout::NC);
                layerData->setPrecision(Precision::FP32);
            } else {
                throw std::runtime_error("Unknown type of input layer layout. "
                                         "Expected either 4 or 2 dimensional inputs");
            }
        }

        auto outputInfo = network.getOutputsInfo();
        auto outputName = outputInfo.begin()->first;
        for (auto outputBlobsIt = outputInfo.begin(); outputBlobsIt != outputInfo.end(); ++outputBlobsIt) {
            auto layerName = outputBlobsIt->first;
            auto layerData = outputBlobsIt->second;
            auto layerDataDims = layerData->getTensorDesc().getDims();

            std::vector<unsigned long> layerDims(layerDataDims.data(), layerDataDims.data() + layerDataDims.size());
            outputBlobsDimsInfo[layerName] = layerDims;
            layerData->setPrecision(Precision::FP32);
        }
        // ---------------------------------------------------------------------------------------------------

        // ----------------------------------Set device and device settings-----------------------------------
        std::set<std::string> devices;
        if (!FLAGS_d.empty()) {
            for (std::string& device : parseDevices(FLAGS_d)) {
                std::transform(device.begin(), device.end(), device.begin(), ::toupper);
                devices.insert(device);
            }
        }
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
        // ---------------------------------------------------------------------------------------------------

        // --------------------------------------Load network to device---------------------------------------
        ExecutableNetwork executableNetwork;
        executableNetwork = ie.LoadNetwork(network, FLAGS_d);
        // ---------------------------------------------------------------------------------------------------

        // ----------------------------Try to set optimal number of infer requests----------------------------
        if (FLAGS_nireq == 0) {
            std::string key = METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS);
            try {
                FLAGS_nireq = executableNetwork.GetMetric(key).as<unsigned int>();
            } catch (const details::InferenceEngineException& ex) {
                THROW_IE_EXCEPTION
                        << "Every device used with the imagenet_classification_demo should "
                        << "support OPTIMAL_NUMBER_OF_INFER_REQUESTS ExecutableNetwork metric. "
                        << "Failed to query the metric for the " << FLAGS_d << " with error:" << ex.what();
            }
        }
        // ---------------------------------------------------------------------------------------------------

        // ---------------------------------------Create infer request----------------------------------------
        std::vector<InferRequest> inferRequests;
        int infReqNum = FLAGS_nireq;
        for (int infReqID = 0; infReqID < infReqNum; ++infReqID) {
            inferRequests.push_back(executableNetwork.CreateInferRequest());
        }
        // ---------------------------------------------------------------------------------------------------
        
        // ----------------------------------------Create output info-----------------------------------------
        Presenter presenter(FLAGS_u, 0);
        int width;
        int height;
        std::vector<std::string> gridMatRowsCols = split(FLAGS_res, 'x');        
        if (gridMatRowsCols.size() != 2) {
            throw std::runtime_error("The value of GridMat resolution flag is not valid.");
        } else {
            width = std::stoi(gridMatRowsCols[0]);
            height = std::stoi(gridMatRowsCols[1]);
        }
        GridMat gridMat = GridMat(inputImagesCount, presenter, cv::Size(width, height));

        // image, predicted label, isPredictionRight
        std::list<std::tuple<cv::Mat, std::string, bool>> shownImagesInfo;
        // ---------------------------------------------------------------------------------------------------
        
        // -----------------------------Prepare variables and data for main loop------------------------------
        double avgFPS = 0;
        double avgLatency = 0;
        double latencySum = 0;
        unsigned framesNum = 0;
        long long rightPredictionsCount = 0;
        long long totalPredictionsCount = 0;
        double accuracy;
        bool isTestMode = true;
        int delay = 1;
        char key = 0;
        std::size_t nextImageIndex = 0;
        long long startTickCount = cv::getTickCount();
        std::condition_variable condVar;
        std::mutex mutex;
        
        std::queue<IRInfo> emptyInferRequests;
        std::queue<IRInfo> completedInferRequests;
        for (std::size_t i = 0; i < inferRequests.size(); i++) {
            emptyInferRequests.push(IRInfo(inferRequests[i], std::vector<IRInfo::IRImage>()));
        }

        auto startTime = std::chrono::system_clock::now();
        auto currentTime = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsedSeconds = currentTime - startTime;
        // ---------------------------------------------------------------------------------------------------
        


        // ===================================================================================================
        do {
            if (irCallbackException) std::rethrow_exception(irCallbackException);

            if (isTestMode && elapsedSeconds.count() >= 10) {
                isTestMode = false;

                gridMat = GridMat(inputImagesCount, presenter, cv::Size(width, height), cv::Size(16, 9), avgFPS);
                cv::Size gridMatSize = gridMat.getSize();
                delay = ((FLAGS_delay == -1)
                         ? static_cast<int>(avgFPS / (gridMatSize.width * gridMatSize.height) * 1000)
                         : FLAGS_delay);
            }

            std::unique_lock<std::mutex> lock(mutex);
        
            if (!completedInferRequests.empty()) {
                auto completedIRInfo = completedInferRequests.front();
                completedInferRequests.pop();
                mutex.unlock();
                emptyInferRequests.push(IRInfo(completedIRInfo.ir, std::vector<IRInfo::IRImage>()));

                // -----------------------------------Process output blobs------------------------------------
                std::vector<unsigned> rightClasses = {};
                for (size_t i = 0; i < completedIRInfo.images.size(); i++) {
                    rightClasses.push_back(completedIRInfo.images[i].rightClass);
                }

                ClassificationResult res(completedIRInfo.ir.GetBlob(outputName), FLAGS_b, FLAGS_nt);
                std::vector<unsigned> results = res.topResults(FLAGS_nt, *completedIRInfo.ir.GetBlob(outputName));
                std::vector<std::string> predictedLabels = {};
                for (size_t i = 0; i < FLAGS_b; i++) {
                    bool isPredictionRight = false;
                    std::string firstPredicted = labels[results[FLAGS_nt*i]];
                    std::string predictedRight;
                    for (size_t j = 0; j < FLAGS_nt; j++) {
                        unsigned predictedClass = results[FLAGS_nt*i+j];
                        if (predictedClass == rightClasses[i]) {
                            isPredictionRight = true;
                            predictedRight = labels[predictedClass];
                            break;
                        }
                    }
                    if (isPredictionRight) {
                        predictedLabels.push_back(predictedRight);
                        rightPredictionsCount++;
                    }
                    else {
                        predictedLabels.push_back(firstPredicted);
                    }
                    totalPredictionsCount++;

                    shownImagesInfo.push_back(
                        std::make_tuple(completedIRInfo.images[i].mat, predictedLabels[i], isPredictionRight));
                }

                framesNum += FLAGS_b;
                gridMat.listUpdate(shownImagesInfo);

                avgFPS = 1. / (((cv::getTickCount() - startTickCount) / cv::getTickFrequency()) / framesNum);
                
                std::vector<long long> processingEndTimes;
                if (!FLAGS_no_show && FLAGS_delay != -1) {
                    cv::imshow("main", gridMat.getMat(processingEndTimes));
                    key = static_cast<char>(cv::waitKey(delay));
                }

                for (size_t i = 0; i < FLAGS_b; i++) {
                    latencySum += ((processingEndTimes[i] - completedIRInfo.images[i].startTime)
                                   / cv::getTickFrequency());
                }
                avgLatency = latencySum / framesNum;
                accuracy = static_cast<double>(rightPredictionsCount) / totalPredictionsCount;
                gridMat.textUpdate(avgFPS, avgLatency, accuracy, isTestMode, presenter);
            }
            else if (!emptyInferRequests.empty()) {
                mutex.unlock();
                cv::Mat nextImage = inputImages[nextImageIndex];
                long long IRstartTime = cv::getTickCount();
                resizeImage(nextImage, modelInputResolution);
                emptyInferRequests.front().images.push_back(
                    IRInfo::IRImage(nextImage,
                                    std::stoul(classIndexes[nextImageIndex]),
                                    IRstartTime));
                nextImageIndex++;
                if (nextImageIndex == inputImagesCount) {
                    nextImageIndex = 0;
                }
                if (emptyInferRequests.front().images.size() == FLAGS_b) {
                    auto emptyIR = emptyInferRequests.front();

                    emptyIR.ir.SetCompletionCallback(
                        InferRequestCallback(
                            emptyIR.ir,
                            emptyIR.images,
                            completedInferRequests,
                            mutex,
                            condVar
                        )
                    );
                    
                    auto inputBlob = emptyIR.ir.GetBlob(inputBlobName);
                    for (unsigned i = 0; i < FLAGS_b; i++) {
                        matU8ToBlob<uint8_t>(emptyIR.images[i].mat, inputBlob, i);
                    }
                    emptyIR.ir.StartAsync();
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

        std::cout << "-------------------------------------" << std::endl;
        std::cout << "Overall FPS: " << avgFPS << std::endl;
        std::cout << "Latency: " << avgLatency << std::endl;
        std::cout << "Accuracy: " << accuracy << std::endl;
        std::cout << presenter.reportMeans() << std::endl;

        if (!FLAGS_no_show) {
            cv::destroyWindow("main");
        }
        // ===================================================================================================
        


        // ------------------------------------Wait for all infer requests------------------------------------
        for (InferRequest& inferRequest : inferRequests)
            inferRequest.Wait(IInferRequest::WaitMode::RESULT_READY);
        // ---------------------------------------------------------------------------------------------------
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
