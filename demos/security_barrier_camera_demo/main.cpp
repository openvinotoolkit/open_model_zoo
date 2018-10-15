/*
// Copyright (c) 2018 Intel Corporation
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
* \brief The entry point for the Inference Engine interactive_Vehicle_detection demo application
* \file security_barrier_camera_demo/main.cpp
* \example security_barrier_camera_demo/main.cpp
*/
#include <gflags/gflags.h>
#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <algorithm>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#include <inference_engine.hpp>

#include <samples/common.hpp>
#include "security_barrier_camera.hpp"
#include "mkldnn/mkldnn_extension_ptr.hpp"
#include <ext_list.hpp>

#include <opencv2/opencv.hpp>

using namespace InferenceEngine;

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }

    std::cout << "[ INFO ] Parsing input parameters" << std::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    return true;
}

// -------------------------Generic routines for detection networks-------------------------------------------------

struct BaseDetection {
    ExecutableNetwork net;
    InferencePlugin plugin;
    InferRequest request;
    std::string & commandLineFlag;
    std::string topoName;
    Blob::Ptr inputBlob;
    std::string inputName;
    std::string outputName;

    BaseDetection(std::string &commandLineFlag, std::string topoName)
            : commandLineFlag(commandLineFlag), topoName(topoName) {}

    ExecutableNetwork  * operator ->() {
        return &net;
    }
    virtual CNNNetwork read()  = 0;

    virtual void setRoiBlob(const Blob::Ptr &roiBlob) {
        if (!enabled())
            return;
        if (!request)
            request = net.CreateInferRequest();

        request.SetBlob(inputName, roiBlob);
    }

    virtual void enqueue(const cv::Mat &frame) {
        if (!enabled())
            return;
        if (!request)
            request = net.CreateInferRequest();

        if (FLAGS_auto_resize) {
            inputBlob = wrapMat2Blob(frame);
            request.SetBlob(inputName, inputBlob);
        } else {
            inputBlob = request.GetBlob(inputName);
            matU8ToBlob<uint8_t>(frame, inputBlob);
        }
    }

    virtual void submitRequest() {
        if (!enabled() || !request) return;
        request.StartAsync();
    }

    virtual void wait() {
        if (!enabled()|| !request) return;
        request.Wait(IInferRequest::WaitMode::RESULT_READY);
    }
    mutable bool enablingChecked = false;
    mutable bool _enabled = false;

    bool enabled() const  {
        if (!enablingChecked) {
            _enabled = !commandLineFlag.empty();
            if (!_enabled) {
                std::cout << "[ INFO ] " << topoName << " detection DISABLED" << std::endl;
            }
            enablingChecked = true;
        }
        return _enabled;
    }
};

struct VehicleDetection : BaseDetection{
    int maxProposalCount;
    int objectSize;
    float width = 0;
    float height = 0;
    bool resultsFetched = false;

    struct Result {
        int label;
        float confidence;
        cv::Rect location;
    };

    std::vector<Result> results;

    void submitRequest() override {
        resultsFetched = false;
        results.clear();
        BaseDetection::submitRequest();
    }

    void setRoiBlob(const Blob::Ptr &frameBlob) override {
        height = frameBlob->getTensorDesc().getDims()[2];
        width = frameBlob->getTensorDesc().getDims()[3];
        BaseDetection::setRoiBlob(frameBlob);
    }

    void enqueue(const cv::Mat &frame) override {
        height = frame.rows;
        width = frame.cols;
        BaseDetection::enqueue(frame);
    }

    VehicleDetection() : BaseDetection(FLAGS_m, "Vehicle") {}
    CNNNetwork read() override {
        std::cout << "[ INFO ] Loading network files for VehicleDetection" << std::endl;
        CNNNetReader netReader;
        /** Read network model **/
        netReader.ReadNetwork(FLAGS_m);
        /** Set batch size to 1 **/
        std::cout << "[ INFO ] Batch size is forced to  1" << std::endl;
        netReader.getNetwork().setBatchSize(1);
        /** Extract model name and load it's weights **/
        std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
        netReader.ReadWeights(binFileName);
        // -----------------------------------------------------------------------------------------------------

        /** SSD-based network should have one input and one output **/
        // ---------------------------Check inputs ------------------------------------------------------
        std::cout << "[ INFO ] Checking Vehicle Detection inputs" << std::endl;
        InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
        if (inputInfo.size() != 1) {
            throw std::logic_error("Vehicle Detection network should have only one input");
        }
        InputInfo::Ptr& inputInfoFirst = inputInfo.begin()->second;
        inputInfoFirst->setInputPrecision(Precision::U8);

        if (FLAGS_auto_resize) {
            inputInfoFirst->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
            inputInfoFirst->getInputData()->setLayout(Layout::NHWC);
        } else {
            inputInfoFirst->getInputData()->setLayout(Layout::NCHW);
        }
        inputName = inputInfo.begin()->first;
        // -----------------------------------------------------------------------------------------------------

        // ---------------------------Check outputs ------------------------------------------------------
        std::cout << "[ INFO ] Checking Vehicle Detection outputs" << std::endl;
        OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
        if (outputInfo.size() != 1) {
            throw std::logic_error("Vehicle Detection network should have only one output");
        }
        DataPtr& _output = outputInfo.begin()->second;
        const SizeVector outputDims = _output->getTensorDesc().getDims();
        outputName = outputInfo.begin()->first;
        maxProposalCount = outputDims[2];
        objectSize = outputDims[3];
        if (objectSize != 7) {
            throw std::logic_error("Output should have 7 as a last dimension");
        }
        if (outputDims.size() != 4) {
            throw std::logic_error("Incorrect output dimensions for SSD");
        }
        _output->setPrecision(Precision::FP32);
        _output->setLayout(Layout::NCHW);

        std::cout << "[ INFO ] Loading Vehicle Detection model to the "<< FLAGS_d << " plugin" << std::endl;
        return netReader.getNetwork();
    }

    void fetchResults() {
        if (!enabled()) return;
        results.clear();
        if (resultsFetched) return;
        resultsFetched = true;
        const float *detections = request.GetBlob(outputName)->buffer().as<float *>();
        // pretty much regular SSD post-processing
        for (int i = 0; i < maxProposalCount; i++) {
            float image_id = detections[i * objectSize + 0];  // in case of batch
            Result r;
            r.label = static_cast<int>(detections[i * objectSize + 1]);
            r.confidence = detections[i * objectSize + 2];
            if (r.confidence <= FLAGS_t) {
                continue;
            }

            r.location.x = detections[i * objectSize + 3] * width;
            r.location.y = detections[i * objectSize + 4] * height;
            r.location.width = detections[i * objectSize + 5] * width - r.location.x;
            r.location.height = detections[i * objectSize + 6] * height - r.location.y;

            if (image_id < 0) {  // indicates end of detections
                break;
            }
            if (FLAGS_r) {
                std::cout << "[" << i << "," << r.label << "] element, prob = " << r.confidence <<
                          "    (" << r.location.x << "," << r.location.y << ")-(" << r.location.width << ","
                          << r.location.height << ")"
                          << ((r.confidence > FLAGS_t) ? " WILL BE RENDERED!" : "") << std::endl;
            }
            results.push_back(r);
        }
    }
};

struct VehicleAttribsDetection : BaseDetection {
    std::string outputNameForType;
    std::string outputNameForColor;

    VehicleAttribsDetection() : BaseDetection(FLAGS_m_va, "Vehicle Attribs") {}

    struct Attributes { std::string type; std::string color;};
    Attributes GetAttributes() {
        static const std::string colors[] = {
                "white", "gray", "yellow", "red", "green", "blue", "black"
        };
        static const std::string types[] = {
                "car", "van", "truck", "bus"
        };

        // 7 possible colors for each vehicle and we should select the one with the maximum probability
        auto colorsValues = request.GetBlob(outputNameForColor)->buffer().as<float*>();
        // 4 possible types for each vehicle and we should select the one with the maximum probability
        auto typesValues  = request.GetBlob(outputNameForType)->buffer().as<float*>();

        const auto color_id = std::max_element(colorsValues, colorsValues + 7) - colorsValues;
        const auto type_id =  std::max_element(typesValues,  typesValues  + 4) - typesValues;
        return {types[type_id], colors[color_id]};
    }

    CNNNetwork read() override {
        std::cout << "[ INFO ] Loading network files for VehicleAttribs" << std::endl;
        CNNNetReader netReader;
        /** Read network model **/
        netReader.ReadNetwork(FLAGS_m_va);
        netReader.getNetwork().setBatchSize(1);
        std::cout << "[ INFO ] Batch size is forced to 1 for Vehicle Attribs" << std::endl;

        /** Extract model name and load it's weights **/
        std::string binFileName = fileNameNoExt(FLAGS_m_va) + ".bin";
        netReader.ReadWeights(binFileName);
        // -----------------------------------------------------------------------------------------------------

        /** Vehicle Attribs network should have one input two outputs **/
        // ---------------------------Check inputs ------------------------------------------------------
        std::cout << "[ INFO ] Checking VehicleAttribs inputs" << std::endl;
        InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
        if (inputInfo.size() != 1) {
            throw std::logic_error("Vehicle Attribs topology should have only one input");
        }
        InputInfo::Ptr& inputInfoFirst = inputInfo.begin()->second;
        inputInfoFirst->setInputPrecision(Precision::U8);
        if (FLAGS_auto_resize) {
            inputInfoFirst->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
            inputInfoFirst->getInputData()->setLayout(Layout::NHWC);
        } else {
            inputInfoFirst->getInputData()->setLayout(Layout::NCHW);
        }
        inputName = inputInfo.begin()->first;
        // -----------------------------------------------------------------------------------------------------

        // ---------------------------Check outputs ------------------------------------------------------
        std::cout << "[ INFO ] Checking Vehicle Attribs outputs" << std::endl;
        OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
        if (outputInfo.size() != 2) {
            throw std::logic_error("Vehicle Attribs Network expects networks having two outputs");
        }
        auto it = outputInfo.begin();
        outputNameForColor = (it++)->second->name;  // color is the first output
        outputNameForType = (it++)->second->name;  // type is the second output

        std::cout << "[ INFO ] Loading Vehicle Attribs model to the "<< FLAGS_d_va << " plugin" << std::endl;
        _enabled = true;
        return netReader.getNetwork();
    }
};

struct LPRDetection : BaseDetection {
    std::string inputSeqName;
    const int maxSequenceSizePerPlate = 88;
    LPRDetection() : BaseDetection(FLAGS_m_lpr, "License Plate Recognition") {}

    void fillSeqBlob() {
        Blob::Ptr seqBlob = request.GetBlob(inputSeqName);
        // second input is sequence, which is some relic from the training
        // it should have the leading 0.0f and rest 1.0f
        float* blob_data = seqBlob->buffer().as<float*>();
        blob_data[0] = 0.0f;
        std::fill(blob_data + 1, blob_data + maxSequenceSizePerPlate, 1.0f);
    }

    void setRoiBlob(const Blob::Ptr &lprBlob) {
        BaseDetection::setRoiBlob(lprBlob);
        fillSeqBlob();
    }

    void enqueue(const cv::Mat &frame) override {
        BaseDetection::enqueue(frame);
        fillSeqBlob();
    }

    std::string GetLicencePlateText() {
        static std::vector<std::string> items = {
                "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                "<Anhui>", "<Beijing>", "<Chongqing>", "<Fujian>",
                "<Gansu>", "<Guangdong>", "<Guangxi>", "<Guizhou>",
                "<Hainan>", "<Hebei>", "<Heilongjiang>", "<Henan>",
                "<HongKong>", "<Hubei>", "<Hunan>", "<InnerMongolia>",
                "<Jiangsu>", "<Jiangxi>", "<Jilin>", "<Liaoning>",
                "<Macau>", "<Ningxia>", "<Qinghai>", "<Shaanxi>",
                "<Shandong>", "<Shanghai>", "<Shanxi>", "<Sichuan>",
                "<Tianjin>", "<Tibet>", "<Xinjiang>", "<Yunnan>",
                "<Zhejiang>", "<police>",
                "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
                "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
                "U", "V", "W", "X", "Y", "Z"
        };
        // up to 88 items per license plate, ended with "-1"
        const auto data = request.GetBlob(outputName)->buffer().as<float*>();
        std::string result;
        for (int i = 0; i < maxSequenceSizePerPlate; i++) {
            if (data[i] == -1)
                break;
            result += items[data[i]];
        }
        return result;
    }
    CNNNetwork read() override {
        std::cout << "[ INFO ] Loading network files for Licence Plate Recognition (LPR)" << std::endl;
        CNNNetReader netReader;
        /** Read network model **/
        netReader.ReadNetwork(FLAGS_m_lpr);
        std::cout << "[ INFO ] Batch size is forced to  1 for LPR Network" << std::endl;
        netReader.getNetwork().setBatchSize(1);
        /** Extract model name and load it's weights **/
        std::string binFileName = fileNameNoExt(FLAGS_m_lpr) + ".bin";
        netReader.ReadWeights(binFileName);

        /** LPR network should have 2 inputs (and second is just a stub) and one output **/
        // ---------------------------Check inputs ------------------------------------------------------
        std::cout << "[ INFO ] Checking LPR Network inputs" << std::endl;
        InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
        if (inputInfo.size() != 2) {
            throw std::logic_error("LPR should have 2 inputs");
        }
        InputInfo::Ptr& inputInfoFirst = inputInfo.begin()->second;
        inputInfoFirst->setInputPrecision(Precision::U8);
        if (FLAGS_auto_resize) {
            inputInfoFirst->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
            inputInfoFirst->getInputData()->setLayout(Layout::NHWC);
        } else {
            inputInfoFirst->getInputData()->setLayout(Layout::NCHW);
        }
        inputName = inputInfo.begin()->first;
        auto sequenceInput = (++inputInfo.begin());
        inputSeqName = sequenceInput->first;
        if (sequenceInput->second->getTensorDesc().getDims()[0] != maxSequenceSizePerPlate) {
            throw std::logic_error("LPR post-processing assumes certain maximum sequences");
        }
        // -----------------------------------------------------------------------------------------------------

        // ---------------------------Check outputs ------------------------------------------------------
        std::cout << "[ INFO ] Checking LPR Network outputs" << std::endl;
        OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
        if (outputInfo.size() != 1) {
            throw std::logic_error("LPR should have 1 output");
        }
        outputName = outputInfo.begin()->first;
        std::cout << "[ INFO ] Loading LPR model to the "<< FLAGS_d_lpr << " plugin" << std::endl;

        _enabled = true;
        return netReader.getNetwork();
    }
};

struct Load {
    BaseDetection& detector;
    explicit Load(BaseDetection& detector) : detector(detector) { }

    void into(InferencePlugin & plg) const {
        if (detector.enabled()) {
            detector.net = plg.LoadNetwork(detector.read(), {});
            detector.plugin = plg;
        }
    }
};

int main(int argc, char *argv[]) {
    try {
        /** This demo covers 3 certain topologies and cannot be generalized **/
        std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        std::cout << "[ INFO ] Reading input" << std::endl;
        cv::Mat frame = cv::imread(FLAGS_i, cv::IMREAD_COLOR);
        const bool isVideo = frame.empty();
        cv::VideoCapture cap;
        if (isVideo && !(FLAGS_i == "cam" ? cap.open(0) : cap.open(FLAGS_i))) {
            throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
        }
        const size_t width  = isVideo ? (size_t) cap.get(CV_CAP_PROP_FRAME_WIDTH) : frame.size().width;
        const size_t height = isVideo ? (size_t) cap.get(CV_CAP_PROP_FRAME_HEIGHT) : frame.size().height;
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Load Plugin for inference engine -------------------------------------
        std::map<std::string, InferencePlugin> pluginsForNetworks;
        std::vector<std::string> pluginNames = {
                FLAGS_d, FLAGS_d_va, FLAGS_d_lpr
        };
        LPRDetection LPR;
        VehicleDetection VehicleDetection;
        VehicleAttribsDetection VehicleAttribs;

        for (auto && flag : pluginNames) {
            if (flag == "") continue;
            auto i = pluginsForNetworks.find(flag);
            if (i != pluginsForNetworks.end()) {
                continue;
            }
            std::cout << "[ INFO ] Loading plugin " << flag << std::endl;
            InferencePlugin plugin = PluginDispatcher({"../../../lib/intel64", ""}).getPluginByDevice(flag);

            /** Printing plugin version **/
            printPluginVersion(plugin, std::cout);

            if ((flag.find("CPU") != std::string::npos)) {
                /** Load default extensions lib for the CPU plugin (e.g. SSD's DetectionOutput)**/
                plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
                if (!FLAGS_l.empty()) {
                    // Any user-specified CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
                    auto extension_ptr = make_so_pointer<MKLDNNPlugin::IMKLDNNExtension>(FLAGS_l);
                    plugin.AddExtension(std::static_pointer_cast<IExtension>(extension_ptr));
                }
            }

            if ((flag.find("GPU") != std::string::npos) && !FLAGS_c.empty()) {
                // Load any user-specified clDNN Extensions
                plugin.SetConfig({ { PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c } });
            }
            pluginsForNetworks[flag] = plugin;
        }

        /** Per layer metrics **/
        if (FLAGS_pc) {
            for (auto && plugin : pluginsForNetworks) {
                plugin.second.SetConfig({{PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES}});
            }
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 2. Read IR models and load them to plugins ------------------------------
        Load(VehicleDetection).into(pluginsForNetworks[FLAGS_d]);
        Load(VehicleAttribs).into(pluginsForNetworks[FLAGS_d_va]);
        Load(LPR).into(pluginsForNetworks[FLAGS_d_lpr]);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Do inference ---------------------------------------------------------
        Blob::Ptr frameBlob;  // this blob includes pixel data from each frame
        ROI cropRoi;  // cropped image coordinates
        Blob::Ptr roiBlob;  // This blob contains data from cropped image (vehicle or license plate)

        std::cout << "[ INFO ] Start inference " << std::endl;
        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

        /** Start inference & calc performance **/
        auto total_t0 = std::chrono::high_resolution_clock::now();
        do {
            // get and enqueue the next frame (in case of video)
            if (isVideo && !cap.read(frame)) {
                if (frame.empty())
                    break;  // end of video file
                throw std::logic_error("Failed to get frame from cv::VideoCapture");
            }

            if (FLAGS_auto_resize) {
                // just wrap Mat object with Blob::Ptr without additional memory allocation
                frameBlob = wrapMat2Blob(frame);
                VehicleDetection.setRoiBlob(frameBlob);
            } else {
                VehicleDetection.enqueue(frame);
            }
            // ----------------------------Run Vehicle detection inference------------------------------------------
            auto t0 = std::chrono::high_resolution_clock::now();
            VehicleDetection.submitRequest();
            VehicleDetection.wait();
            auto t1 = std::chrono::high_resolution_clock::now();
            ms detection = std::chrono::duration_cast<ms>(t1 - t0);
            // parse inference results internally (e.g. apply a threshold, etc)
            VehicleDetection.fetchResults();
            // -----------------------------------------------------------------------------------------------------

            // ----------------------------Process the results down to the pipeline---------------------------------
            ms AttribsNetworkTime(0), LPRNetworktime(0);
            int AttribsInferred = 0,  LPRInferred = 0;
            for (auto && result : VehicleDetection.results) {
                std::string attrColor = "";
                std::string attrType = "";
                std::string lprStr = "";

                if (result.label == 1) {  // vehicle
                    if (VehicleAttribs.enabled()) {
                        // ----------------------------Run vehicle attribute ----------------
                        if (FLAGS_auto_resize) {
                            cropRoi.posX = (result.location.x < 0) ? 0 : result.location.x;
                            cropRoi.posY = (result.location.y < 0) ? 0 : result.location.y;
                            cropRoi.sizeX = std::min((size_t) result.location.width, width - cropRoi.posX);
                            cropRoi.sizeY = std::min((size_t) result.location.height, height - cropRoi.posY);
                            roiBlob = make_shared_blob(frameBlob, cropRoi);
                            VehicleAttribs.setRoiBlob(roiBlob);
                        } else {
                            // To crop ROI manually and allocate required memory (cv::Mat) again
                            auto clippedRect = result.location & cv::Rect(0, 0, width, height);
                            cv::Mat vehicle = frame(clippedRect);
                            VehicleAttribs.enqueue(vehicle);
                        }

                        t0 = std::chrono::high_resolution_clock::now();
                        VehicleAttribs.submitRequest();
                        VehicleAttribs.wait();
                        t1 = std::chrono::high_resolution_clock::now();
                        AttribsNetworkTime += std::chrono::duration_cast<ms>(t1 - t0);
                        AttribsInferred++;

                        attrColor = VehicleAttribs.GetAttributes().color;
                        attrType = VehicleAttribs.GetAttributes().type;
                    }
                    cv::rectangle(frame, result.location, cv::Scalar(0, 255, 0), 2);
                } else {
                    if (LPR.enabled()) {  // licence plate
                        // ----------------------------Run License Plate Recognition ----------------
                        // expanding a bounding box a bit, better for the license plate recognition
                        result.location.x -= 5;
                        result.location.y -= 5;
                        result.location.width += 10;
                        result.location.height += 10;
                        if (FLAGS_auto_resize) {
                            cropRoi.posX = (result.location.x < 0) ? 0 : result.location.x;
                            cropRoi.posY = (result.location.y < 0) ? 0 : result.location.y;
                            cropRoi.sizeX = std::min((size_t) result.location.width, width - cropRoi.posX);
                            cropRoi.sizeY = std::min((size_t) result.location.height, height - cropRoi.posY);
                            roiBlob = make_shared_blob(frameBlob, cropRoi);
                            LPR.setRoiBlob(roiBlob);
                        } else {
                            // To crop ROI manually and allocate required memory (cv::Mat) again
                            auto clippedRect = result.location & cv::Rect(0, 0, width, height);
                            cv::Mat plate = frame(clippedRect);
                            LPR.enqueue(plate);
                        }

                        t0 = std::chrono::high_resolution_clock::now();
                        LPR.submitRequest();
                        LPR.wait();
                        t1 = std::chrono::high_resolution_clock::now();
                        LPRNetworktime += std::chrono::duration_cast<ms>(t1 - t0);
                        LPRInferred++;
                        lprStr = LPR.GetLicencePlateText();
                    }
                    cv::rectangle(frame, result.location, cv::Scalar(0, 0, 255), 2);
                }
                // ----------------------------Process outputs-----------------------------------------------------
                if (!attrColor.empty() && !attrType.empty()) {
                    cv::putText(frame,
                                attrColor,
                                cv::Point2f(result.location.x, result.location.y + 15),
                                cv::FONT_HERSHEY_COMPLEX_SMALL,
                                0.8,
                                cv::Scalar(255, 255, 255));
                    cv::putText(frame,
                                attrType,
                                cv::Point2f(result.location.x, result.location.y + 30),
                                cv::FONT_HERSHEY_COMPLEX_SMALL,
                                0.8,
                                cv::Scalar(255, 255, 255));
                    if (FLAGS_r) {
                        std::cout << "Vehicle Attributes results:" << attrColor << ";" << attrType << std::endl;
                    }
                }
                if (!lprStr.empty()) {
                    cv::putText(frame,
                                lprStr,
                                cv::Point2f(result.location.x, result.location.y + result.location.height + 15),
                                cv::FONT_HERSHEY_COMPLEX_SMALL,
                                0.8,
                                cv::Scalar(0, 0, 255));
                    if (FLAGS_r) {
                        std::cout << "License Plate Recognition results:" << lprStr << std::endl;
                    }
                }
            }

            // ----------------------------Execution statistics -----------------------------------------------------
            std::ostringstream out;
            out << "Vehicle detection time  : " << std::fixed << std::setprecision(2) << detection.count()
                << " ms ("
                << 1000.f / detection.count() << " fps)";
            cv::putText(frame, out.str(), cv::Point2f(0, 20), cv::FONT_HERSHEY_TRIPLEX, 0.5,
                        cv::Scalar(255, 0, 0));
            if (VehicleDetection.results.size()) {
                if (VehicleAttribs.enabled() && AttribsInferred) {
                    float average_time = AttribsNetworkTime.count() / AttribsInferred;
                    out.str("");
                    out << "Vehicle Attribs time (averaged over " << AttribsInferred << " detections) :" << std::fixed
                        << std::setprecision(2) << average_time << " ms " << "(" << 1000.f / average_time << " fps)";
                    cv::putText(frame, out.str(), cv::Point2f(0, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                                cv::Scalar(255, 0, 0));
                }
                if (LPR.enabled() && LPRInferred) {
                    float average_time = LPRNetworktime.count() / LPRInferred;
                    out.str("");
                    out << "LPR time (averaged over " << LPRInferred << " detections) :" << std::fixed
                        << std::setprecision(2) << average_time << " ms " << "(" << 1000.f / average_time << " fps)";
                    cv::putText(frame, out.str(), cv::Point2f(0, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                                cv::Scalar(255, 0, 0));
                }
            }

            if (!FLAGS_no_show) {
                cv::imshow("Detection results", frame);
            }
            // for still images wait until any key is pressed, for video 1 ms is enough per frame
            const int key = cv::waitKey(isVideo ? 1 : 0);
            if (27 == key)  // Esc
                break;
        } while (isVideo);
        // -----------------------------------------------------------------------------------------------------
        auto total_t1 = std::chrono::high_resolution_clock::now();
        ms total = std::chrono::duration_cast<ms>(total_t1 - total_t0);
        std::cout << std::endl << "Total execution time: " << total.count() << std::endl << std::endl;

        /** Show performace results **/
        if (FLAGS_pc) {
            for (auto && plugin : pluginsForNetworks) {
                std::cout << "[ INFO ] Performance counts for " << plugin.first << " plugin";
                printPerformanceCountsPlugin(plugin.second, std::cout);
            }
        }
    }
    catch (const std::exception& error) {
        std::cerr << "[ ERROR ] " << error.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
        return 1;
    }

    std::cout << "[ INFO ] Execution successful" << std::endl;
    return 0;
}
