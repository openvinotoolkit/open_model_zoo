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
#include <cstdio>
#include <functional>
#include <atomic>

#include <inference_engine.hpp>

#include <ie_iextension.h>

#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>
#include <samples/ocv_common.hpp>

#include <vpu/vpu_plugin_config.hpp>
#include <cldnn/cldnn_config.hpp>

#include "imagenet_classification_demo.hpp"
#include "grid_mat.hpp"

using namespace InferenceEngine;

struct InferRequestInfo {
    struct InferRequestImage {
        cv::Mat mat;
        unsigned correctClass;
        std::chrono::time_point<std::chrono::steady_clock> startTime;
    };

    InferRequest &inferRequest;
    std::vector<InferRequestImage> images;
};

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------------Parsing and validation of input args----------------------------------
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
    if (FLAGS_labels.empty()) {
        throw std::logic_error("Parameter -labels is not set");
    }
    if (FLAGS_nt < 1) {
        throw std::runtime_error("Parameter -nt must be >= 1");
    }

    return true;
}

cv::Mat resizeImage(cv::Mat& image, int modelInputResolution) {
    double scale = static_cast<double>(modelInputResolution) / std::min(image.cols, image.rows);

    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(), scale, scale);

    cv::Rect imgROI;
    if (resizedImage.cols >= resizedImage.rows) {
        int fromWidth = resizedImage.cols/2 - modelInputResolution/2;
        imgROI = cv::Rect(fromWidth,
                          0,
                          resizedImage.cols - fromWidth,
                          modelInputResolution);
    } else {
        int fromHeight = resizedImage.rows/2 - modelInputResolution/2;
        imgROI = cv::Rect(0,
                          fromHeight,
                          modelInputResolution,
                          resizedImage.rows - fromHeight);
    }

    return resizedImage(imgROI);
}

std::vector<std::vector<unsigned>> topResults(Blob& inputBlob, unsigned numTop) {
    using currentBlobType = PrecisionTrait<Precision::FP32>::value_type;
    TBlob<currentBlobType>& tblob = dynamic_cast<TBlob<currentBlobType>&>(inputBlob);
    size_t batchSize =  tblob.getTensorDesc().getDims()[0];
    numTop = static_cast<unsigned>(std::min<size_t>(size_t(numTop), tblob.size()));
    
    std::vector<std::vector<unsigned>> output(batchSize);
    for (size_t i = 0; i < batchSize; i++) {
        size_t offset = i * (tblob.size() / batchSize);
        currentBlobType *batchData = tblob.data();
        batchData += offset;
        std::vector<unsigned> indices(tblob.size() / batchSize);
        std::iota(std::begin(indices), std::end(indices), 0);
        std::partial_sort(std::begin(indices), std::begin(indices) + numTop, std::end(indices),
                          [&batchData](unsigned l, unsigned r) {
                              return batchData[l] > batchData[r];
                          });

        indices.resize(numTop);
        output[i] = std::move(indices);
    }
    
    return output;
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
        size_t inputImagesCount = imageNames.size();
        if (inputImagesCount == 0) throw std::runtime_error("No images provided");
        std::sort(imageNames.begin(), imageNames.end());
        for (size_t i = 0; i < inputImagesCount; i++) {
            const std::string& name = imageNames[i];
            const cv::Mat& tmpImage = cv::imread(name);
            if (tmpImage.data == nullptr) {
                std::cerr << "Could not read image " << name << '\n';
                imageNames.erase(imageNames.begin() + i);
                inputImagesCount--;
                i--;
            } else {
                inputImages.push_back(tmpImage);
                imageNames[i] = name.substr(name.rfind('/') + 1);
            }
        }
        // ---------------------------------------------------------------------------------------------------

        // ----------------------------------------Read image classes-----------------------------------------
        std::vector<unsigned> classIndices;
        if (!FLAGS_gt.empty()) {
            std::map<std::string, std::string> classIndicesMap;
            std::ifstream inputGtFile(FLAGS_gt);
            if (!inputGtFile.is_open()) throw std::runtime_error("Can't open the ground truth file.");
            while (true) {
                std::string imageName;
                std::string classIndex;
                inputGtFile >> imageName >> classIndex;
                if (inputGtFile.eof()) break;
                classIndicesMap.insert({imageName.substr(imageName.rfind('/') + 1), classIndex});
            }
            for (size_t i = 0; i < inputImagesCount; i++) {
                auto imageSearchResult = classIndicesMap.find(imageNames[i]);
                if (imageSearchResult != classIndicesMap.end()) {
                    classIndices.push_back(static_cast<unsigned>(std::stoul(imageSearchResult->second)));
                } else {
                    throw std::runtime_error("No class specified for image " + imageNames[i]);
                }
            }
        } else {
            classIndices.resize(inputImages.size());
            std::fill(classIndices.begin(), classIndices.end(), 0);
        }
        // ---------------------------------------------------------------------------------------------------

        // --------------------------------------------Read labels--------------------------------------------
        std::vector<std::string> labels;
        std::ifstream inputLabelsFile(FLAGS_labels);
        if (!inputLabelsFile.is_open()) throw std::runtime_error("Can't open the labels file.");
        std::string labelsLine;
        while (std::getline(inputLabelsFile, labelsLine)) {
            labels.push_back(labelsLine.substr(labelsLine.find(' ') + 1,
                                               labelsLine.find(',') - (labelsLine.find(' ') + 1)));
        }
        // ---------------------------------------------------------------------------------------------------

        // -------------------------------------------Read network--------------------------------------------
        Core ie;
        CNNNetwork network = ie.ReadNetwork(FLAGS_m);
        // ---------------------------------------------------------------------------------------------------

        // --------------------------------------Configure model input----------------------------------------
        auto inputShapes = network.getInputShapes();
        if (inputShapes.size() != 1) {
            throw std::logic_error("The network should have only one input.");
        }

        std::string inputBlobName = inputShapes.begin()->first;
        SizeVector& inputShape = inputShapes.begin()->second;
        if (inputShape.size() != 4) {
            throw std::logic_error("Model input has incorrect number of dimensions. Must be 4.");
        }
        if (inputShape[1] != 3) {
            throw std::logic_error("Model input has incorrect number of color channels."
                                   " Expected 3, got " + std::to_string(inputShape[1]) + ".");
        }
        if (inputShape[2] != inputShape[3]) {
            throw std::logic_error("Model input has incorrect image shape. Must be NxN square."
                                   " Got " + std::to_string(inputShape[2]) + 
                                   "x" + std::to_string(inputShape[3]) + ".");
        }
        int modelInputResolution = inputShape[2];

        inputShape[0] = FLAGS_b;
        network.reshape(inputShapes);

        auto inputLayerData = network.getInputsInfo().begin()->second;
        inputLayerData->setLayout(Layout::NCHW);
        inputLayerData->setPrecision(Precision::U8);
        // ---------------------------------------------------------------------------------------------------

        // --------------------------------------Configure model output---------------------------------------
        auto outputInfo = network.getOutputsInfo();
        if (outputInfo.size() != 1) {
            throw std::logic_error("The network should have only one output.");
        }

        auto outputName = outputInfo.begin()->first;
        auto outputLayerData = outputInfo.begin()->second;
        auto layerDataDims = outputLayerData->getTensorDesc().getDims();
        if (layerDataDims.size() != 2 && layerDataDims.size() != 4) {
            throw std::logic_error("Incorrect number of dimensions in model output layer. Must be 2 or 4.");
        }
        if (layerDataDims[1] == labels.size() + 1) {
            labels.insert(labels.begin(), "other");
            for (size_t i = 0; i < classIndices.size(); i++) {
                classIndices[i]++;
            }
        } else if (layerDataDims[1] != labels.size() || layerDataDims[0] != FLAGS_b) {
            throw std::logic_error("Incorrect size of model output layer. Must be BatchSize x NumberOfClasses.");
        }

        outputLayerData->setPrecision(Precision::FP32);
        // ---------------------------------------------------------------------------------------------------

        // ----------------------------------Set device and device settings-----------------------------------
        std::set<std::string> devices;
        for (std::string& device : parseDevices(FLAGS_d)) {
            std::transform(device.begin(), device.end(), device.begin(), ::toupper);
            devices.insert(device);
        }
        std::map<std::string, unsigned> deviceNstreams = parseValuePerDevice(devices, FLAGS_nstreams);
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
                                (deviceNstreams.count(device) > 0 ? std::to_string(deviceNstreams.at(device)) :
                                                                                 "CPU_THROUGHPUT_AUTO") }}, device);
                deviceNstreams[device] = std::stoi(
                    ie.GetConfig(device, CONFIG_KEY(CPU_THROUGHPUT_STREAMS)).as<std::string>());
            } else if (device == "GPU") {
                ie.SetConfig({{ CONFIG_KEY(GPU_THROUGHPUT_STREAMS),
                                (deviceNstreams.count(device) > 0 ? std::to_string(deviceNstreams.at(device)) :
                                                                         "GPU_THROUGHPUT_AUTO") }}, device);
                deviceNstreams[device] = std::stoi(
                    ie.GetConfig(device, CONFIG_KEY(GPU_THROUGHPUT_STREAMS)).as<std::string>());

                if ((FLAGS_d.find("MULTI") != std::string::npos) &&
                    (FLAGS_d.find("CPU") != std::string::npos)) {
                    // multi-device execution with the CPU + GPU performs best with GPU throttling hint,
                    // which releases another CPU thread (that is otherwise used by the GPU driver for active polling)
                    ie.SetConfig({{ CLDNN_CONFIG_KEY(PLUGIN_THROTTLE), "1" }}, "GPU");
                }
            }
        }
        // ---------------------------------------------------------------------------------------------------

        // --------------------------------------Load network to device---------------------------------------
        ExecutableNetwork executableNetwork = ie.LoadNetwork(network, FLAGS_d);
        // ---------------------------------------------------------------------------------------------------

        // ----------------------------Try to set optimal number of infer requests----------------------------
        unsigned nireq = FLAGS_nireq;
        if (nireq == 0) {
            std::string key = METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS);
            try {
                nireq = executableNetwork.GetMetric(key).as<unsigned int>();
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
        for (unsigned infReqId = 0; infReqId < nireq; ++infReqId) {
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
        GridMat gridMat = GridMat(presenter, cv::Size(width, height));
        // ---------------------------------------------------------------------------------------------------
        
        // -----------------------------Prepare variables and data for main loop------------------------------
        double avgFPS = 0;
        double avgLatency = 0;
        double latencySum = 0;
        unsigned framesNum = 0;
        long long correctPredictionsCount = 0;
        double accuracy = 0;
        bool isTestMode = true;
        char key = 0;
        std::size_t nextImageIndex = 0;
        auto startTickCount = std::chrono::steady_clock::now();
        std::condition_variable condVar;
        std::mutex mutex;
        std::atomic<bool> hasCallbackException(false);
        std::exception_ptr irCallbackException;
        
        std::queue<InferRequestInfo> emptyInferRequests;
        std::queue<InferRequestInfo> completedInferRequests;
        for (std::size_t i = 0; i < inferRequests.size(); i++) {
            emptyInferRequests.push({inferRequests[i], std::vector<InferRequestInfo::InferRequestImage>()});
        }

        auto startTime = std::chrono::steady_clock::now();
        auto elapsedSeconds = std::chrono::steady_clock::duration{0};
        // ---------------------------------------------------------------------------------------------------
        
        // -------------------------------------Processing infer requests-------------------------------------
        int framesNumOnCalculationStart = 0;
        auto testDuration = std::chrono::seconds{3};
        auto fpsCalculationDuration = std::chrono::seconds{1};
        typedef std::chrono::duration<double, std::chrono::seconds::period> Sec;
        do {
            if (hasCallbackException) std::rethrow_exception(irCallbackException);

            if (elapsedSeconds >= testDuration - fpsCalculationDuration && framesNumOnCalculationStart == 0) {
                framesNumOnCalculationStart = framesNum;
            }
            if (isTestMode && elapsedSeconds >= testDuration) {
                isTestMode = false;
                gridMat = GridMat(presenter, cv::Size(width, height), cv::Size(16, 9),
                                  (framesNum - framesNumOnCalculationStart) / std::chrono::duration_cast<Sec>(
                                    fpsCalculationDuration).count());
                startTickCount = std::chrono::steady_clock::now();
                framesNum = 0;
                latencySum = 0;
                correctPredictionsCount = 0;
                accuracy = 0;
            }

            std::unique_lock<std::mutex> lock(mutex);
            if (!completedInferRequests.empty()) {
                auto completedInferRequestInfo = completedInferRequests.front();
                completedInferRequests.pop();
                lock.unlock();
                emptyInferRequests.push({completedInferRequestInfo.inferRequest, 
                                         std::vector<InferRequestInfo::InferRequestImage>()});

                std::vector<unsigned> correctClasses = {};
                for (size_t i = 0; i < completedInferRequestInfo.images.size(); i++) {
                    correctClasses.push_back(completedInferRequestInfo.images[i].correctClass);
                }

                std::vector<std::vector<unsigned>> results = topResults(
                    *completedInferRequestInfo.inferRequest.GetBlob(outputName), FLAGS_nt);
                std::vector<std::string> predictedLabels = {};
                ImageInfoList shownImagesInfo;
                for (size_t i = 0; i < FLAGS_b; i++) {
                    PredictionResult predictionResult = PredictionResult::Incorrect;
                    if (!FLAGS_gt.empty()) {
                        for (size_t j = 0; j < FLAGS_nt; j++) {
                            unsigned predictedClass = results[i][j];
                            if (predictedClass == correctClasses[i]) {
                                predictionResult = PredictionResult::Correct;
                                predictedLabels.push_back(labels[predictedClass]);
                                correctPredictionsCount++;
                                break;
                            }
                        }
                    } else {
                        predictionResult = PredictionResult::Unknown;
                    }

                    if (predictionResult != PredictionResult::Correct) {
                        predictedLabels.push_back(labels[results[i][0]]);
                    }

                    shownImagesInfo.push_back(
                        std::make_tuple(completedInferRequestInfo.images[i].mat, predictedLabels[i], predictionResult));
                }

                framesNum += FLAGS_b;
                
                avgFPS = framesNum / std::chrono::duration_cast<Sec>(
                    std::chrono::steady_clock::now() - startTickCount).count();
                gridMat.updateMat(shownImagesInfo);
                auto processingEndTime = std::chrono::steady_clock::now();
                for (const auto & image : completedInferRequestInfo.images) {
                    latencySum += (1. * std::chrono::duration_cast<Sec>(processingEndTime - image.startTime).count());
                }
                avgLatency = latencySum / framesNum;
                accuracy = static_cast<double>(correctPredictionsCount) / framesNum;
                gridMat.textUpdate(avgFPS, avgLatency, accuracy, isTestMode, !FLAGS_gt.empty(), presenter);
                
                if (!FLAGS_no_show) {
                    cv::imshow("imagenet_classification_demo", gridMat.outImg);
                    key = static_cast<char>(cv::waitKey(1));
                    presenter.handleKey(key);
                }
            } else if (!emptyInferRequests.empty()) {
                lock.unlock();
                auto inferRequestStartTime = std::chrono::steady_clock::now();
                cv::Mat nextImage = resizeImage(inputImages[nextImageIndex], modelInputResolution);
                emptyInferRequests.front().images.push_back(
                                                    {nextImage,
                                                     classIndices[nextImageIndex],
                                                     inferRequestStartTime});
                nextImageIndex++;
                if (nextImageIndex == inputImagesCount) {
                    nextImageIndex = 0;
                }
                if (emptyInferRequests.front().images.size() == FLAGS_b) {
                    auto emptyInferRequest = emptyInferRequests.front();
                    emptyInferRequests.pop();

                    emptyInferRequest.inferRequest.SetCompletionCallback([emptyInferRequest,
                                                      &completedInferRequests,
                                                      &mutex,
                                                      &condVar,
                                                      &hasCallbackException,
                                                      &irCallbackException] {
                        try {
                            std::lock_guard<std::mutex> callback_lock(mutex);
                            completedInferRequests.push({emptyInferRequest.inferRequest, emptyInferRequest.images});
                        }
                        catch(...) {
                            if (!hasCallbackException) {
                                irCallbackException = std::current_exception();
                                hasCallbackException = true;
                            }
                        }
                        condVar.notify_one();
                    });
                    
                    auto inputBlob = emptyInferRequest.inferRequest.GetBlob(inputBlobName);
                    for (unsigned i = 0; i < FLAGS_b; i++) {
                        matU8ToBlob<uint8_t>(emptyInferRequest.images[i].mat, inputBlob, i);
                    }
                    emptyInferRequest.inferRequest.StartAsync();
                }
            }
            
            lock.lock();
            while (emptyInferRequests.empty() && completedInferRequests.empty()) {
                condVar.wait(lock);
            }
            lock.unlock();

            elapsedSeconds = std::chrono::steady_clock::now() - startTime;
        } while (key != 27 && key != 'q' && key != 'Q'
                 && (FLAGS_time == -1 || elapsedSeconds < std::chrono::seconds{FLAGS_time}));

        std::cout << "FPS: " << avgFPS << std::endl;
        std::cout << "Latency: " << avgLatency << std::endl;
        if (!FLAGS_gt.empty()) {
            std::cout << "Accuracy (top " << FLAGS_nt << "): " << accuracy << std::endl;
        }
        std::cout << presenter.reportMeans() << std::endl;
        // ---------------------------------------------------------------------------------------------------
        
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
