// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief The entry point for the Inference Engine object_detection demo application
* \file object_detection_demo_ssd_async/main.cpp
* \example object_detection_demo_ssd_async/main.cpp
*/

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <vector>
#include <deque>
#include <string>
#include <algorithm>

#include <ngraph/ngraph.hpp>

#include <monitors/presenter.h>
#include <samples/ocv_common.hpp>
#include <samples/args_helper.hpp>
#include <cldnn/cldnn_config.hpp>

#include "object_detection_demo_ssd_async.hpp"
#include "ssd_async.h"

using namespace InferenceEngine;

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

void frameToBlob(const cv::Mat& frame,
                 InferRequest::Ptr& inferRequest,
                 const std::string& inputName) {
    if (FLAGS_auto_resize) {
        /* Just set input blob containing read image. Resize and layout conversion will be done automatically */
        inferRequest->SetBlob(inputName, wrapMat2Blob(frame));
    } else {
        /* Resize and copy data from the image to the input blob */
        Blob::Ptr frameBlob = inferRequest->GetBlob(inputName);
        matU8ToBlob<uint8_t>(frame, frameBlob);
    }
}

void putHighlightedText(cv::Mat& img,
                        const std::string& text,
                        cv::Point org,
                        int fontFace,
                        double fontScale,
                        cv::Scalar color,
                        int thickness) {
    cv::putText(img, text, org, fontFace, fontScale, cv::Scalar(255, 255, 255), thickness + 1); // white border
    cv::putText(img, text, org, fontFace, fontScale, color, thickness);
}

int main(int argc, char *argv[]) {
    try {
        /** This demo covers certain topology and cannot be generalized for any object detection **/
        std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        slog::info << "Reading input" << slog::endl;
        cv::VideoCapture cap;
        if (!((FLAGS_i == "cam") ? cap.open(0) : cap.open(FLAGS_i.c_str()))) {
            throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
        }
        const size_t width  = (size_t)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        const size_t height = (size_t)cap.get(cv::CAP_PROP_FRAME_HEIGHT);

        // read input (video) frame
        cv::Mat curr_frame;  cap >> curr_frame;
        cv::Mat next_frame;

        if (!cap.grab()) {
            throw std::logic_error("This demo supports only video (or camera) inputs !!! "
                                   "Failed getting next frame from the " + FLAGS_i);
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Load inference engine ------------------------------------------------
        slog::info << "Loading Inference Engine" << slog::endl;
        Core ie;

        slog::info << "Device info: " << slog::endl;
        std::cout << ie.GetVersions(FLAGS_d);

        /** Load extensions for the plugin **/
        if (!FLAGS_l.empty()) {
            // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
            IExtensionPtr extension_ptr = make_so_pointer<IExtension>(FLAGS_l.c_str());
            ie.AddExtension(extension_ptr, "CPU");
        }
        if (!FLAGS_c.empty()) {
            // clDNN Extensions are loaded from an .xml description and OpenCL kernel files
            ie.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}}, "GPU");
        }

        /** Per layer metrics **/
        if (FLAGS_pc) {
            ie.SetConfig({ { PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES } });
        }

        std::map<std::string, std::string> userSpecifiedConfig;
        std::map<std::string, std::string> minLatencyConfig;

        std::set<std::string> devices;
        for (const std::string& device : parseDevices(FLAGS_d)) {
            devices.insert(device);
        }
        std::map<std::string, unsigned> deviceNstreams = parseValuePerDevice(devices, FLAGS_nstreams);
        for (auto & device : devices) {
            if (device == "CPU") {  // CPU supports a few special performance-oriented keys
                // limit threading for CPU portion of inference
                if (FLAGS_nthreads != 0)
                    userSpecifiedConfig.insert({ CONFIG_KEY(CPU_THREADS_NUM), std::to_string(FLAGS_nthreads) });

                if (FLAGS_d.find("MULTI") != std::string::npos
                    && devices.find("GPU") != devices.end()) {
                    userSpecifiedConfig.insert({ CONFIG_KEY(CPU_BIND_THREAD), CONFIG_VALUE(NO) });
                } else {
                    // pin threads for CPU portion of inference
                    userSpecifiedConfig.insert({ CONFIG_KEY(CPU_BIND_THREAD), CONFIG_VALUE(YES) });
                }

                // for CPU execution, more throughput-oriented execution via streams
                userSpecifiedConfig.insert({ CONFIG_KEY(CPU_THROUGHPUT_STREAMS),
                                (deviceNstreams.count(device) > 0 ? std::to_string(deviceNstreams.at(device))
                                                                  : CONFIG_VALUE(CPU_THROUGHPUT_AUTO)) });

                minLatencyConfig.insert({ CONFIG_KEY(CPU_THROUGHPUT_STREAMS), "1" });
            } else if (device == "GPU") {
                userSpecifiedConfig.insert({ CONFIG_KEY(GPU_THROUGHPUT_STREAMS),
                                (deviceNstreams.count(device) > 0 ? std::to_string(deviceNstreams.at(device))
                                                                  : CONFIG_VALUE(GPU_THROUGHPUT_AUTO)) });

                minLatencyConfig.insert({ CONFIG_KEY(GPU_THROUGHPUT_STREAMS), "1" });

                if (FLAGS_d.find("MULTI") != std::string::npos
                    && devices.find("CPU") != devices.end()) {
                    // multi-device execution with the CPU + GPU performs best with GPU throttling hint,
                    // which releases another CPU thread (that is otherwise used by the GPU driver for active polling)
                    userSpecifiedConfig.insert({ CLDNN_CONFIG_KEY(PLUGIN_THROTTLE), "1" });
                }
            }
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 2. Read IR Generated by ModelOptimizer (.xml and .bin files) ------------
        slog::info << "Loading network files" << slog::endl;
        /** Read network model **/
        ssd_async  ssd( ie, FLAGS_m);
        /** Set batch size to 1 **/
        slog::info << "Batch size is forced to 1." << slog::endl;
        (ssd.get_cnnNetwork()).setBatchSize(1);
        /** Read labels (if any)**/
        std::vector<std::string> labels;
        if (!FLAGS_labels.empty()) {
            std::ifstream inputFile(FLAGS_labels);
            std::string label;
            while (std::getline(inputFile, label)) {
                labels.push_back(label);
            }
            if (labels.empty())
                throw std::logic_error("File empty or not found: " + FLAGS_labels);
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Configure input & output ---------------------------------------------
        // --------------------------- Prepare input blobs -----------------------------------------------------
        ssd.PreparingInputBlobs();

        // --------------------------- Prepare output blobs -----------------------------------------------------
        ssd.PreparingOutputBlobs(labels);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Loading model to the device ------------------------------------------
        slog::info << "Loading model to the device" << slog::endl;
        ExecutableNetwork userSpecifiedExecNetwork = ie.LoadNetwork(ssd.get_cnnNetwork(), FLAGS_d, userSpecifiedConfig);
        ExecutableNetwork minLatencyExecNetwork = ie.LoadNetwork(ssd.get_cnnNetwork(), FLAGS_d, minLatencyConfig);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Create infer requests ------------------------------------------------
        

        InferRequest::Ptr minLatencyInferRequest = ssd.CreateInferRequest(userSpecifiedExecNetwork, minLatencyExecNetwork, FLAGS_nireq);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Init variables -------------------------------------------------------
        typedef std::chrono::duration<double, std::chrono::milliseconds::period> Ms;
        typedef std::chrono::duration<double, std::chrono::seconds::period> Sec;

        enum class ExecutionMode {USER_SPECIFIED, MIN_LATENCY};
        
        struct RequestResult {
            cv::Mat frame;
            Blob::Ptr output;
            std::chrono::steady_clock::time_point startTime;
            ExecutionMode frameMode;
        };

        struct ModeInfo {
           int framesCount = 0;
           std::chrono::steady_clock::duration latencySum;
           std::chrono::steady_clock::time_point lastStartTime;
           std::chrono::steady_clock::time_point lastEndTime;
        };

        auto totalT0 = std::chrono::steady_clock::now();
        ExecutionMode currentMode = ExecutionMode::USER_SPECIFIED;
        std::deque<InferRequest::Ptr> emptyRequests;
        if (currentMode == ExecutionMode::USER_SPECIFIED) {
            emptyRequests.assign(userSpecifiedInferRequests.begin(), userSpecifiedInferRequests.end());
        } else emptyRequests = {minLatencyInferRequest};

        std::unordered_map<int, RequestResult> completedRequestResults;
        int nextFrameId = 0;
        int nextFrameIdToShow = 0;
        std::exception_ptr callbackException = nullptr;
        std::mutex mutex;
        std::condition_variable condVar;
        std::map<ExecutionMode, ModeInfo> modeInfo = {{ExecutionMode::USER_SPECIFIED, ModeInfo()},
                                                      {ExecutionMode::MIN_LATENCY, ModeInfo()}};

        cv::Size graphSize{static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH) / 4), 60};
        Presenter presenter(FLAGS_u, static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)) - graphSize.height - 10,
                            graphSize);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 7. Do inference ---------------------------------------------------------
        std::cout << "To close the application, press 'CTRL+C' here or switch to the output window and "
                     "press ESC or 'q' key" << std::endl;
        std::cout << "To switch between min_latency/user_specified modes, press TAB key in the output window" 
                  << std::endl;

        modeInfo[currentMode].lastStartTime = std::chrono::steady_clock::now();
        while (true) {
            {
                std::lock_guard<std::mutex> lock(mutex);

                if (callbackException) std::rethrow_exception(callbackException);
                if (!cap.isOpened()
                    && completedRequestResults.empty()
                    && ((currentMode == ExecutionMode::USER_SPECIFIED && emptyRequests.size() == FLAGS_nireq)
                        || (currentMode == ExecutionMode::MIN_LATENCY && emptyRequests.size() == 1))) break;
            }

            RequestResult requestResult;
            {
                std::lock_guard<std::mutex> lock(mutex);

                auto requestResultItr = completedRequestResults.find(nextFrameIdToShow);
                if (requestResultItr != completedRequestResults.end()) {
                    requestResult = std::move(requestResultItr->second);
                    completedRequestResults.erase(requestResultItr);
                };
            }

            if (requestResult.output) {
                LockedMemory<const void> outputMapped = as<MemoryBlob>(requestResult.output)->rmap();
                const float *detections = outputMapped.as<float*>();

                nextFrameIdToShow++;
                if (requestResult.frameMode == currentMode) {
                    modeInfo[currentMode].framesCount += 1;
                }

                ssd.ProcessingOutputBlobs(detections, width, height, labels, FLAGS_r, FLAGS_t);

                presenter.drawGraphs(requestResult.frame);

                auto currentTime = std::chrono::steady_clock::now();
                std::ostringstream out;
                out << (currentMode == ExecutionMode::USER_SPECIFIED ? "USER SPECIFIED" : "MIN LATENCY")
                    << " mode (press Tab to switch)";
                putHighlightedText(requestResult.frame, out.str(), cv::Point2f(10, 30), cv::FONT_HERSHEY_TRIPLEX, 0.75, 
                                   cv::Scalar(10, 10, 200), 2);
                out.str("");
                out << "FPS: " << std::fixed << std::setprecision(1) << modeInfo[currentMode].framesCount / 
                    std::chrono::duration_cast<Sec>(currentTime - modeInfo[currentMode].lastStartTime).count();
                putHighlightedText(requestResult.frame, out.str(), cv::Point2f(10, 60), cv::FONT_HERSHEY_TRIPLEX, 0.75, 
                                   cv::Scalar(10, 200, 10), 2);
                out.str("");
                modeInfo[currentMode].latencySum += (currentTime - requestResult.startTime);
                out << "Latency: " << std::fixed << std::setprecision(1) << (std::chrono::duration_cast<Ms>(
                        modeInfo[currentMode].latencySum).count() / modeInfo[currentMode].framesCount)
                    << " ms";
                putHighlightedText(requestResult.frame, out.str(), cv::Point2f(10, 90), cv::FONT_HERSHEY_TRIPLEX, 0.75, 
                                   cv::Scalar(200, 10, 10), 2);

                if (!FLAGS_no_show) {
                    cv::imshow("Detection Results", requestResult.frame);
                    
                    const int key = cv::waitKey(1);

                    if (27 == key || 'q' == key || 'Q' == key) {  // Esc
                        break;
                    } else if (9 == key) {  // Tab
                        ExecutionMode prevMode = currentMode;
                        currentMode = (currentMode == ExecutionMode::USER_SPECIFIED ? ExecutionMode::MIN_LATENCY
                                                                                    : ExecutionMode::USER_SPECIFIED);
                        if (prevMode == ExecutionMode::USER_SPECIFIED) {
                            for (const auto& request: userSpecifiedInferRequests) {
                                request->Wait(IInferRequest::WaitMode::RESULT_READY);
                            }
                        } else minLatencyInferRequest->Wait(IInferRequest::WaitMode::RESULT_READY);
                        
                        {
                            std::lock_guard<std::mutex> lock(mutex);

                            if (currentMode == ExecutionMode::USER_SPECIFIED) {
                                emptyRequests.assign(userSpecifiedInferRequests.begin(),
                                                     userSpecifiedInferRequests.end());
                            } else emptyRequests = {minLatencyInferRequest};
                        }

                        modeInfo[currentMode] = ModeInfo();
                        auto switchTime = std::chrono::steady_clock::now();
                        modeInfo[prevMode].lastEndTime = switchTime;
                        modeInfo[currentMode].lastStartTime = switchTime;
                    } else {
                        presenter.handleKey(key);
                    }
                }

                continue;
            }

            {
                std::unique_lock<std::mutex> lock(mutex);

                if (emptyRequests.empty() || !cap.isOpened()) {
                    while (callbackException == nullptr && emptyRequests.empty() && completedRequestResults.empty()) {
                        condVar.wait(lock);
                    }

                    continue;
                }
            }
            
            auto startTime = std::chrono::steady_clock::now();
                
            cv::Mat frame;
            if (!cap.read(frame)) {
                if (frame.empty()) {
                    if (FLAGS_loop_input) {
                        cap.open((FLAGS_i == "cam") ? 0 : FLAGS_i.c_str());
                    } else cap.release();
                    continue;
                } else {
                    throw std::logic_error("Failed to get frame from cv::VideoCapture");
                }
            }

            InferRequest::Ptr request;
            {
                std::lock_guard<std::mutex> lock(mutex);

                request = std::move(emptyRequests.front());
                emptyRequests.pop_front();
            }
            frameToBlob(frame, request, ssd.get_imageInputName());
                
            ExecutionMode frameMode = currentMode;
            request->SetCompletionCallback([&mutex,
                                            &completedRequestResults,
                                            nextFrameId,
                                            frame,
                                            request,
                                            &ssd.get_outputName(),
                                            startTime,
                                            frameMode,
                                            &emptyRequests,
                                            &callbackException,
                                            &condVar] {
                {
                    std::lock_guard<std::mutex> callbackLock(mutex);
                
                    try {
                        completedRequestResults.insert(
                            std::pair<int, RequestResult>(nextFrameId, RequestResult{
                                frame,
                                std::make_shared<TBlob<float>>(*as<TBlob<float>>(request->GetBlob(ssd.get_outputName()))),
                                startTime,
                                frameMode
                            }));
                        
                        emptyRequests.push_back(std::move(request));
                    }
                    catch (...) {
                        if (!callbackException) {
                            callbackException = std::current_exception();
                        }
                    }
                }
                condVar.notify_one();
            });

            request->StartAsync();
            nextFrameId++;
        }
        // -----------------------------------------------------------------------------------------------------
        
        // --------------------------- 8. Report metrics -------------------------------------------------------
        slog::info << slog::endl << "Metric reports:" << slog::endl;

        auto totalT1 = std::chrono::steady_clock::now();
        Ms totalTime = std::chrono::duration_cast<Ms>(totalT1 - totalT0);
        std::cout << std::endl << "Total Inference time: " << totalTime.count() << std::endl;

        /** Show performace results **/
        if (FLAGS_pc) {
            if (currentMode == ExecutionMode::USER_SPECIFIED) {
                for (const auto& request: userSpecifiedInferRequests) {
                    printPerformanceCounts(*request, std::cout, getFullDeviceName(ie, FLAGS_d));
                }
            } else printPerformanceCounts(*minLatencyInferRequest, std::cout, getFullDeviceName(ie, FLAGS_d));
        }

        std::chrono::steady_clock::time_point endTime;
        if (modeInfo[ExecutionMode::USER_SPECIFIED].framesCount) {
            std::cout << std::endl;
            std::cout << "USER_SPECIFIED mode:" << std::endl;
            endTime = (modeInfo[ExecutionMode::USER_SPECIFIED].lastEndTime.time_since_epoch().count() != 0)
                      ? modeInfo[ExecutionMode::USER_SPECIFIED].lastEndTime
                      : totalT1;
            std::cout << "FPS: " << std::fixed << std::setprecision(1)
                << modeInfo[ExecutionMode::USER_SPECIFIED].framesCount / std::chrono::duration_cast<Sec>(
                    endTime - modeInfo[ExecutionMode::USER_SPECIFIED].lastStartTime).count() << std::endl;
            std::cout << "Latency: " << std::fixed << std::setprecision(1)
                << ((std::chrono::duration_cast<Ms>(modeInfo[ExecutionMode::USER_SPECIFIED].latencySum).count() /
                     modeInfo[ExecutionMode::USER_SPECIFIED].framesCount)) << " ms" << std::endl;
        }

        if (modeInfo[ExecutionMode::MIN_LATENCY].framesCount) {
            std::cout << std::endl;
            std::cout << "MIN_LATENCY mode:" << std::endl;
            endTime = (modeInfo[ExecutionMode::MIN_LATENCY].lastEndTime.time_since_epoch().count() != 0)
                      ? modeInfo[ExecutionMode::MIN_LATENCY].lastEndTime
                      : totalT1;
            std::cout << "FPS: " << std::fixed << std::setprecision(1)
                << modeInfo[ExecutionMode::MIN_LATENCY].framesCount / std::chrono::duration_cast<Sec>(
                    endTime - modeInfo[ExecutionMode::MIN_LATENCY].lastStartTime).count() << std::endl;
            std::cout << "Latency: " << std::fixed << std::setprecision(1)
                << ((std::chrono::duration_cast<Ms>(modeInfo[ExecutionMode::MIN_LATENCY].latencySum).count() /
                     modeInfo[ExecutionMode::MIN_LATENCY].framesCount)) << " ms" << std::endl;
        }

        std::cout << std::endl << presenter.reportMeans() << '\n';
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 9. Wait for running Infer Requests --------------------------------------
        if (currentMode == ExecutionMode::USER_SPECIFIED) {
            for (const auto& request: userSpecifiedInferRequests) {
                request->Wait(IInferRequest::WaitMode::RESULT_READY);
            }
        } else minLatencyInferRequest->Wait(IInferRequest::WaitMode::RESULT_READY);
        // -----------------------------------------------------------------------------------------------------
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
