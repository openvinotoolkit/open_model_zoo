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

#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>
#include <samples/ocv_common.hpp>

#include <cldnn/cldnn_config.hpp>

#include "classification_demo.hpp"
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

cv::Mat resizeImage(const cv::Mat& image, int modelInputResolution) {
    double scale = static_cast<double>(modelInputResolution) / std::min(image.cols, image.rows);

    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(), scale, scale);

    cv::Rect imgROI;
    if (resizedImage.cols >= resizedImage.rows) {
        int fromWidth = resizedImage.cols/2 - modelInputResolution/2;
        imgROI = cv::Rect(fromWidth, 0, modelInputResolution, modelInputResolution);
    } else {
        int fromHeight = resizedImage.rows/2 - modelInputResolution/2;
        imgROI = cv::Rect(0, fromHeight, modelInputResolution, modelInputResolution);
    }

    return resizedImage(imgROI);
}

std::vector<std::vector<unsigned>> topResults(Blob& inputBlob, unsigned numTop) {
    TBlob<float>& tblob = dynamic_cast<TBlob<float>&>(inputBlob);
    size_t batchSize =  tblob.getTensorDesc().getDims()[0];
    numTop = static_cast<unsigned>(std::min<size_t>(size_t(numTop), tblob.size()));
    
    std::vector<std::vector<unsigned>> output(batchSize);
    for (size_t i = 0; i < batchSize; i++) {
        size_t offset = i * (tblob.size() / batchSize);
        float *batchData = tblob.data() + offset;
        std::vector<unsigned> indices(tblob.size() / batchSize);
        std::iota(std::begin(indices), std::end(indices), 0);
        std::partial_sort(std::begin(indices), std::begin(indices) + numTop, std::end(indices),
                          [&batchData](unsigned l, unsigned r) {
                              return batchData[l] > batchData[r];
                          });

        output[i].assign(indices.begin(), indices.begin() + numTop);
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
        if (imageNames.empty()) throw std::runtime_error("No images provided");
        std::sort(imageNames.begin(), imageNames.end());
        for (size_t i = 0; i < imageNames.size(); i++) {
            const std::string& name = imageNames[i];
            const cv::Mat& tmpImage = cv::imread(name);
            if (tmpImage.data == nullptr) {
                std::cerr << "Could not read image " << name << '\n';
                imageNames.erase(imageNames.begin() + i);
                i--;
            } else {
                inputImages.push_back(tmpImage);
                size_t lastSlashIdx = name.find_last_of("/\\");
                if (lastSlashIdx != std::string::npos) {
                    imageNames[i] = name.substr(lastSlashIdx + 1);
                } else {
                    imageNames[i] = name;
                }
            }
        }
        // ---------------------------------------------------------------------------------------------------

        // ----------------------------------------Read image classes-----------------------------------------
        std::vector<unsigned> classIndices;
        if (!FLAGS_gt.empty()) {
            std::map<std::string, unsigned> classIndicesMap;
            std::ifstream inputGtFile(FLAGS_gt);
            if (!inputGtFile.is_open()) throw std::runtime_error("Can't open the ground truth file.");
            
            std::string line;
            while (std::getline(inputGtFile, line))
            {
                size_t separatorIdx = line.find(' ');
                if (separatorIdx == std::string::npos) {
                    throw std::runtime_error("The ground truth file has incorrect format.");
                }
                std::string imagePath = line.substr(0, separatorIdx);
                size_t imagePathEndIdx = imagePath.rfind('/');
                unsigned classIndex = static_cast<unsigned>(std::stoul(line.substr(separatorIdx + 1)));
                if ((imagePathEndIdx != 1 || imagePath[0] != '.') && imagePathEndIdx != std::string::npos) {
                    throw std::runtime_error("The ground truth file has incorrect format.");
                }
                classIndicesMap.insert({imagePath.substr(imagePathEndIdx + 1), classIndex});
            }

            for (size_t i = 0; i < imageNames.size(); i++) {
                auto imageSearchResult = classIndicesMap.find(imageNames[i]);
                if (imageSearchResult != classIndicesMap.end()) {
                    classIndices.push_back(imageSearchResult->second);
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
            size_t labelBeginIdx = labelsLine.find(' ');
            size_t labelEndIdx = labelsLine.find(','); // can be npos when class has only one label
            if (labelBeginIdx == std::string::npos) {
                throw std::runtime_error("The labels file has incorrect format.");
            }
            labels.push_back(labelsLine.substr(labelBeginIdx + 1, labelEndIdx - (labelBeginIdx + 1)));
        }

        for (const auto & classIndex : classIndices) {
            if (classIndex >= labels.size()) {
                throw std::runtime_error("Class index " + std::to_string(classIndex)
                                         + " is outside the range supported by the model.");
            }
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
        }
        if (layerDataDims[1] != labels.size() || layerDataDims[0] != FLAGS_b) {
            throw std::logic_error("Incorrect size of model output layer. Must be BatchSize x NumberOfClasses.");
        }

        outputLayerData->setPrecision(Precision::FP32);
        // ---------------------------------------------------------------------------------------------------

        // ----------------------------------Set device and device settings-----------------------------------
        std::set<std::string> devices;
        for (const std::string& device : parseDevices(FLAGS_d)) {
            devices.insert(device);
        }
        std::map<std::string, unsigned> deviceNstreams = parseValuePerDevice(devices, FLAGS_nstreams);
        for (auto & device : devices) {
            if (device == "CPU") {  // CPU supports a few special performance-oriented keys
                // limit threading for CPU portion of inference
                if (FLAGS_nthreads != 0)
                    ie.SetConfig({{ CONFIG_KEY(CPU_THREADS_NUM), std::to_string(FLAGS_nthreads) }}, device);

                if (FLAGS_d.find("MULTI") != std::string::npos
                    && devices.find("GPU") != devices.end()) {
                    ie.SetConfig({{ CONFIG_KEY(CPU_BIND_THREAD), CONFIG_VALUE(NO) }}, device);
                } else {
                    // pin threads for CPU portion of inference
                    ie.SetConfig({{ CONFIG_KEY(CPU_BIND_THREAD), CONFIG_VALUE(YES) }}, device);
                }

                // for CPU execution, more throughput-oriented execution via streams
                ie.SetConfig({{ CONFIG_KEY(CPU_THROUGHPUT_STREAMS),
                                (deviceNstreams.count(device) > 0 ? std::to_string(deviceNstreams.at(device))
                                                                  : CONFIG_VALUE(CPU_THROUGHPUT_AUTO)) }}, device);
                deviceNstreams[device] = std::stoi(
                    ie.GetConfig(device, CONFIG_KEY(CPU_THROUGHPUT_STREAMS)).as<std::string>());
            } else if (device == "GPU") {
                ie.SetConfig({{ CONFIG_KEY(GPU_THROUGHPUT_STREAMS),
                                (deviceNstreams.count(device) > 0 ? std::to_string(deviceNstreams.at(device))
                                                                  : CONFIG_VALUE(GPU_THROUGHPUT_AUTO)) }}, device);
                deviceNstreams[device] = std::stoi(
                    ie.GetConfig(device, CONFIG_KEY(GPU_THROUGHPUT_STREAMS)).as<std::string>());

                if (FLAGS_d.find("MULTI") != std::string::npos
                    && devices.find("CPU") != devices.end()) {
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
                        << "Every device used with the classification_demo should "
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
        GridMat gridMat(presenter, cv::Size(width, height));
        // ---------------------------------------------------------------------------------------------------
        
        // -----------------------------Prepare variables and data for main loop------------------------------
        typedef std::chrono::duration<double, std::chrono::seconds::period> Sec;
        double avgFPS = 0;
        double avgLatency = 0;
        std::chrono::steady_clock::duration latencySum = std::chrono::steady_clock::duration::zero();
        unsigned framesNum = 0;
        long long correctPredictionsCount = 0;
        double accuracy = 0;
        bool isTestMode = true;
        char key = 0;
        std::size_t nextImageIndex = 0;
        std::condition_variable condVar;
        std::mutex mutex;
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
        do {
            if (irCallbackException) std::rethrow_exception(irCallbackException);

            if (elapsedSeconds >= testDuration - fpsCalculationDuration && framesNumOnCalculationStart == 0) {
                framesNumOnCalculationStart = framesNum;
            }
            if (isTestMode && elapsedSeconds >= testDuration) {
                isTestMode = false;
                gridMat = GridMat(presenter, cv::Size(width, height), cv::Size(16, 9),
                                  (framesNum - framesNumOnCalculationStart) / std::chrono::duration_cast<Sec>(
                                    fpsCalculationDuration).count());
                startTime = std::chrono::steady_clock::now();
                framesNum = 0;
                latencySum = std::chrono::steady_clock::duration::zero();
                correctPredictionsCount = 0;
                accuracy = 0;
            }

            std::unique_ptr<InferRequestInfo> completedInferRequestInfo;
            {
                std::lock_guard<std::mutex> lock(mutex);

                if (!completedInferRequests.empty()) {
                    completedInferRequestInfo.reset(new InferRequestInfo(completedInferRequests.front()));
                    completedInferRequests.pop();
                }
            }
            if (completedInferRequestInfo) {
                emptyInferRequests.push({completedInferRequestInfo->inferRequest, 
                                         std::vector<InferRequestInfo::InferRequestImage>()});

                std::vector<unsigned> correctClasses = {};
                for (size_t i = 0; i < completedInferRequestInfo->images.size(); i++) {
                    correctClasses.push_back(completedInferRequestInfo->images[i].correctClass);
                }

                std::vector<std::vector<unsigned>> results = topResults(
                    *completedInferRequestInfo->inferRequest.GetBlob(outputName), FLAGS_nt);
                std::vector<std::string> predictedLabels = {};
                std::list<LabeledImage> shownImagesInfo;
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
                        LabeledImage{completedInferRequestInfo->images[i].mat, predictedLabels[i], predictionResult});
                }

                framesNum += FLAGS_b;
                
                avgFPS = framesNum / std::chrono::duration_cast<Sec>(
                    std::chrono::steady_clock::now() - startTime).count();
                gridMat.updateMat(shownImagesInfo);
                auto processingEndTime = std::chrono::steady_clock::now();
                for (const auto & image : completedInferRequestInfo->images) {
                    latencySum += processingEndTime - image.startTime;
                }
                avgLatency = std::chrono::duration_cast<Sec>(latencySum).count() / framesNum;
                accuracy = static_cast<double>(correctPredictionsCount) / framesNum;
                gridMat.textUpdate(avgFPS, avgLatency, accuracy, isTestMode, !FLAGS_gt.empty(), presenter);
                
                if (!FLAGS_no_show) {
                    cv::imshow("classification_demo", gridMat.outImg);
                    key = static_cast<char>(cv::waitKey(1));
                    presenter.handleKey(key);
                }

                completedInferRequestInfo.reset();
            } else if (!emptyInferRequests.empty()) {
                auto inferRequestStartTime = std::chrono::steady_clock::now();
                cv::Mat nextImage = resizeImage(inputImages[nextImageIndex], modelInputResolution);
                emptyInferRequests.front().images.push_back(
                                                    {nextImage,
                                                     classIndices[nextImageIndex],
                                                     inferRequestStartTime});
                nextImageIndex++;
                if (nextImageIndex == imageNames.size()) {
                    nextImageIndex = 0;
                }
                if (emptyInferRequests.front().images.size() == FLAGS_b) {
                    auto emptyInferRequest = emptyInferRequests.front();
                    emptyInferRequests.pop();

                    emptyInferRequest.inferRequest.SetCompletionCallback([emptyInferRequest,
                                                      &completedInferRequests,
                                                      &mutex,
                                                      &condVar,
                                                      &irCallbackException] {
                        {
                            std::lock_guard<std::mutex> callbackLock(mutex);
                        
                            try {
                                completedInferRequests.push(emptyInferRequest);
                            }
                            catch(...) {
                                if (!irCallbackException) {
                                    irCallbackException = std::current_exception();
                                }
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

            {
                std::unique_lock<std::mutex> lock(mutex);

                while (!irCallbackException
                       && emptyInferRequests.empty() && completedInferRequests.empty()) {
                    condVar.wait(lock);
                }
            }

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
