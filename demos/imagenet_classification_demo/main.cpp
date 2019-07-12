// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* @brief The entry point the Inference Engine sample application
* @file classification_sample_async/main.cpp
* @example classification_sample_async/main.cpp
*/
#include <fstream>
#include <vector>
#include <queue>
#include <memory>
#include <string>
#include <map>
#include <condition_variable>
#include <mutex>

#include <inference_engine.hpp>

#include <format_reader_ptr.h>

#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>
#include <samples/classification_results.h>
#include <samples/ocv_common.hpp>
#include <samples/common.hpp>

#include <ext_list.hpp>
#include <vpu/vpu_plugin_config.hpp>
#include <cldnn/cldnn_config.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <sys/stat.h>
#include <ext_list.hpp>

#include "classification_sample_async.h"
#include "captures.hpp"
#include "grid_mat.hpp"
#include "my_functor.hpp"

using namespace InferenceEngine;

ConsoleErrorListener error_listener;

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
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

int main(int argc, char *argv[]) {
    try {
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }
            
        std::vector<std::string> imageNames;
        parseInputFilesArguments(imageNames);

        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~read all imgs~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        std::vector<cv::Mat> inputImgs = {};
        for (const auto & i : imageNames) {
            const cv::Mat& tmp = cv::imread(i);
            if (tmp.data == nullptr) {
                std::cerr << "Could not read image " << i << '\n';
            } else {
                inputImgs.push_back(tmp);
            }
        }


    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~????~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(!FLAGS_b)
            FLAGS_b = 1;
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~????~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~read network~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
        Core ie;
        InferenceEngine::CNNNetReader netReader;//
        InferenceEngine::CNNNetwork network;//
        netReader.ReadNetwork(FLAGS_m);
        std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
        netReader.ReadWeights(binFileName);
        network = netReader.getNetwork();

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~init inputBlobName~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        InputsDataMap inputInfo(network.getInputsInfo()); //???
        if (inputInfo.size() != 1) throw std::logic_error("Sample supports topologies with 1 input only");
        std::string inputBlobName = inputInfo.begin()->first;
        Captures captures(inputImgs, inputBlobName, FLAGS_b);


    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~reshape network~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                
        /*
        std::map<std::string, std::vector<unsigned long>> newBlobsDimsInfo;
            auto newBlobDims(inputBlobDims);
            // Fix height and change width to make networkAspectRatio equal to imageAspectRatio
            newBlobDims[3] = static_cast<unsigned long>(newBlobDims[2] * imageAspectRatio);
            newBlobsDimsInfo[inputBlobName] = newBlobDims;
        
        
        if (inputBlobsDimsInfo.size() != newBlobsDimsInfo.size()) {
            throw std::runtime_error("Mismatch in the number of blobs being reshaped");
        }
        */
        auto input_shapes = network.getInputShapes();
        std::string input_name;
        SizeVector input_shape;
        std::tie(input_name, input_shape) = *input_shapes.begin();
        input_shape[0] = FLAGS_b;
        input_shape[2] = input_shapes[input_name][2];
        input_shape[3] = input_shapes[input_name][3];
        input_shapes[input_name] = input_shape;
        std::cout << "Resizing network to the image size = [" << input_shapes[input_name][2] << "x" << input_shapes[input_name][3] << "] "
                  << "with batch = " << FLAGS_b << std::endl;
        network.reshape(input_shapes);

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~????~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        std::map<std::string, std::vector<unsigned long>> inputBlobsDimsInfo;//!!!!!!!!!!!!!!
        std::map<std::string, std::vector<unsigned long>> outputBlobsDimsInfo;//
        //auto inputInfo = network.getInputsInfo();
        for (auto inputBlobsIt = inputInfo.begin(); inputBlobsIt != inputInfo.end(); ++inputBlobsIt) {
            auto layerName = inputBlobsIt->first;
            auto layerData = inputBlobsIt->second;
            auto layerDims = layerData->getTensorDesc().getDims();

            std::vector<unsigned long> layerDims_(layerDims.data(), layerDims.data() + layerDims.size());
            inputBlobsDimsInfo[layerName] = layerDims_;

            if (layerDims.size() == 4) {
                layerData->setLayout(Layout::NCHW);
                layerData->setPrecision(Precision::U8);
            } else if (layerDims.size() == 2) {
                layerData->setLayout(Layout::NC);
                layerData->setPrecision(Precision::FP32);
            } else {
                throw std::runtime_error("Unknow type of input layer layout. Expected either 4 or 2 dimensional inputs");
            }
        }

        // set map of output blob name -- blob dimension pairs
        auto outputInfo = network.getOutputsInfo();
        for (auto outputBlobsIt = outputInfo.begin(); outputBlobsIt != outputInfo.end(); ++outputBlobsIt) {
            auto layerName = outputBlobsIt->first;
            auto layerData = outputBlobsIt->second;
            auto layerDims = layerData->getTensorDesc().getDims();

            std::vector<unsigned long> layerDims_(layerDims.data(), layerDims.data() + layerDims.size());
            outputBlobsDimsInfo[layerName] = layerDims_;
            layerData->setPrecision(Precision::FP32);
        }
        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~????~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Set device and device settings~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        std::set<std::string> devices;
        for (const std::string& netDevices : {FLAGS_d, /*FLAGS_d_va, FLAGS_d_lpr*/}) {
            if (netDevices.empty()) {
                continue;
            }
            for (const std::string& device : parseDevices(netDevices)) {
                devices.insert(device);
            }
        }
        std::map<std::string, uint32_t> device_nstreams = parseValuePerDevice(devices, FLAGS_nstreams);
        for (auto& device : devices) {
            if (device == "CPU") {  // CPU supports few special performance-oriented keys
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
                //if (FLAGS_api == "async")
                ie.SetConfig({{ CONFIG_KEY(CPU_THROUGHPUT_STREAMS),
                                (device_nstreams.count(device) > 0 ? std::to_string(device_nstreams.at(device)) :
                                                                                 "CPU_THROUGHPUT_AUTO") }}, device);
                device_nstreams[device] = std::stoi(ie.GetConfig(device, CONFIG_KEY(CPU_THROUGHPUT_STREAMS)).as<std::string>());
            } else if (device == ("GPU")) {
                //if (FLAGS_api == "async")
                ie.SetConfig({{ CONFIG_KEY(GPU_THROUGHPUT_STREAMS),
                                (device_nstreams.count(device) > 0 ? std::to_string(device_nstreams.at(device)) :
                                                                         "GPU_THROUGHPUT_AUTO") }}, device);
                device_nstreams[device] = std::stoi(ie.GetConfig(device, CONFIG_KEY(GPU_THROUGHPUT_STREAMS)).as<std::string>());

                if ((FLAGS_d.find("MULTI") != std::string::npos) &&
                    (FLAGS_d.find("CPU") != std::string::npos)) {
                    // multi-device execution with the CPU + GPU performs best with GPU trottling hint,
                    // which releases another CPU thread (that is otherwise used by the GPU driver for active polling)
                    ie.SetConfig({{ CLDNN_CONFIG_KEY(PLUGIN_THROTTLE), "1" }}, "GPU");
                }
            } else if (device == "MYRIAD") {
                ie.SetConfig({{ CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_NONE) },
                              { VPU_CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_WARNING) }}, device);
            }
        }

        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~load network to device~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        InferenceEngine::ExecutableNetwork executableNetwork;//
        executableNetwork = ie.LoadNetwork(network, FLAGS_d);

        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~try to set optimal number of infer request~~~~~~~~~~~~~~~~~~~~~`
        if (FLAGS_nireq == 0) {
            std::string key = METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS);
            try {
                FLAGS_nireq = executableNetwork.GetMetric(key).as<unsigned int>();
            } catch (const details::InferenceEngineException& ex) {
                THROW_IE_EXCEPTION
                        << "Every device used with the benchmark_app should "
                        << "support OPTIMAL_NUMBER_OF_INFER_REQUESTS ExecutableNetwork metric. "
                        << "Failed to query the metric for the " << FLAGS_d << " with error:" << ex.what();
            }
        }

        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~create inf req~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        std::vector<InferenceEngine::InferRequest> inferRequests;
        int infReqNum = FLAGS_nireq;
        for(int infReqID = 0; infReqID < infReqNum; ++infReqID)
            inferRequests.push_back(executableNetwork.CreateInferRequest());

       

        std::queue<cv::Mat> showMats;
        
        std::condition_variable condVar;
        std::mutex mutex;
        std::mutex showMutex;

        //curImg is stored in ieWrapper
        int batchSize = FLAGS_b;
        int irFirstIndex = 0;
        
        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~set complition callback~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for (auto it = inferRequests.begin(); it != inferRequests.end(); ++it) {
            it->SetCompletionCallback(
                infReqCallback(*it, mutex, condVar, captures, showMats, irFirstIndex)
            );
            irFirstIndex = (irFirstIndex + batchSize) % inputImgs.size();
        }
        
        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~filling blobs~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        int imgDataSize = inputImgs.size();
        auto blobDims = inputBlobsDimsInfo[inputBlobName];
        if (blobDims.size() != 4) {
            throw std::runtime_error("Input data does not match size of the blob");
        }
        for (auto it = inferRequests.begin(); it != inferRequests.end(); ++it) { 
            //setInputBlob(*it, curPos);    
            auto inputBlob = it->GetBlob(inputBlobName);

            for(int i = 0; i < batchSize; i++) {        
                cv::Mat inputImg = inputImgs.at((captures.curPos+i)%imgDataSize);
                matU8ToBlob<uint8_t>(inputImg, inputBlob, i);
            }
            captures.curPos = (captures.curPos + batchSize) % imgDataSize;
        }

        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~create output info~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        int width;
        int height;
        if(FLAGS_res != "") {
            std::vector<std::string> gmRowsCols = split(FLAGS_res, 'x');        
            if(gmRowsCols.size() != 2) throw std::runtime_error("Input gmsizes key is not valid.");
            else{
                width = std::stoi(gmRowsCols[0]);
                height = std::stoi(gmRowsCols[1]);
            }
        } else {
            width = 1080;
            height =  1920;
        }
        GridMat gridMat = GridMat(cv::Size(width, height));
        cv::namedWindow("main");
        cv::imshow("main", gridMat.getMat());

        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~start async~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        int64 startTime = cv::getTickCount();
        for (auto it = inferRequests.begin(); it != inferRequests.end(); ++it)
            it->StartAsync();


        int64 delay = FLAGS_delay;
        cv::Mat tmpMat;
        while(true) {      
            //double currSPF; 
            double overallSPF;
            {
                std::unique_lock<std::mutex> lock(mutex);
                while(showMats.empty()){   
                    condVar.wait(lock);
                }
                gridMat.listUpdate(showMats);

                //currSPF = (lastInferTime / cv::getTickFrequency()) / batchSize;
                overallSPF = ((cv::getTickCount() - startTime) / cv::getTickFrequency()) / captures.framesNum;
            }
            gridMat.update();
            gridMat.textUpdate(overallSPF);// overallTime is not protected
            cv::imshow("main", gridMat.getMat());
                
            char key = static_cast<char>(cv::waitKey(delay));
            //Press 'Esc' to quit
            if (key == 27){
                captures.quitFlag = true;
                break;
            }
        }
        cv::destroyWindow("main");
        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~wait all irs~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        std::cout<<"wait all irs"<<std::endl;
        for (auto it = inferRequests.begin(); it != inferRequests.end(); ++it)
            it->Wait(IInferRequest::WaitMode::RESULT_READY);
    }
    catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    slog::info << slog::endl << "This sample is an API example, for any performance measurements "
                                "please use the dedicated benchmark_app tool" << slog::endl;
    return 0;
}
