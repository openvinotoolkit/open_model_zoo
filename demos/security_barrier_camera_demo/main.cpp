// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
#include <queue>

#include <inference_engine.hpp>

#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>
#include "security_barrier_camera.hpp"
#include <ie_iextension.h>
#include <ext_list.hpp>

using namespace InferenceEngine;

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }

    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    if (FLAGS_display_resolution.find("x") == std::string::npos) {
        throw std::logic_error("Incorrect format of -displayresolution parameter. Correct format is  \"width x height\". For example \"1920x1080\"");
    }

    return true;
}

// -------------------------Generic routines for detection networks-------------------------------------------------

typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

struct BaseInferRequest {
    InferRequest::Ptr request;
    std::string inputName;
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
    std::chrono::time_point<std::chrono::high_resolution_clock> endTime;
    size_t requestId;

    BaseInferRequest(ExecutableNetwork& net, std::string& inputName) :
        request(net.CreateInferRequestPtr()), inputName(inputName), requestId(0) {
        auto lambda = [&] {
               endTime = std::chrono::high_resolution_clock::now();
            };
        request->SetCompletionCallback(lambda);
    }

    virtual ~BaseInferRequest() {}

    virtual void startAsync() {
        startTime = std::chrono::high_resolution_clock::now();
        request->StartAsync();
    }

    virtual void wait() {
        request->Wait(IInferRequest::WaitMode::RESULT_READY);
    }

    virtual void setBlob(const Blob::Ptr &frameBlob) {
        request->SetBlob(inputName, frameBlob);
    }

    virtual void setImage(const cv::Mat &frame) {
        Blob::Ptr inputBlob;

        if (FLAGS_auto_resize) {
            inputBlob = wrapMat2Blob(frame);
            request->SetBlob(inputName, inputBlob);
        } else {
            inputBlob = request->GetBlob(inputName);
            matU8ToBlob<uint8_t>(frame, inputBlob);
        }
    }

    virtual Blob::Ptr getBlob(const std::string &name) {
        return request->GetBlob(name);
    }

    virtual ms getTime() {
        return std::chrono::duration_cast<ms>(endTime - startTime);
    }

    void setId(size_t id) {
        requestId = id;
    }

    size_t getId() {
        return requestId;
    }

    using Ptr = std::shared_ptr<BaseInferRequest>;
};

struct BaseDetection {
    ExecutableNetwork net;
    InferencePlugin plugin;
    std::string & commandLineFlag;
    std::string topoName;
    std::string inputName;
    std::string outputName;

    BaseDetection(std::string &commandLineFlag, std::string topoName)
            : commandLineFlag(commandLineFlag), topoName(topoName)  {}

    ExecutableNetwork * operator ->() {
        return &net;
    }
    virtual CNNNetwork read()  = 0;

    mutable bool enablingChecked = false;
    mutable bool _enabled = false;

    bool enabled() const  {
        if (!enablingChecked) {
            _enabled = !commandLineFlag.empty();
            if (!_enabled) {
                slog::info << topoName << " detection DISABLED" << slog::endl;
            }
            enablingChecked = true;
        }
        return _enabled;
    }
};

struct VehicleDetectionInferRequest : BaseInferRequest {
    cv::Size srcImageSize;

    VehicleDetectionInferRequest(ExecutableNetwork& net, std::string& inputName) :
        BaseInferRequest(net, inputName), srcImageSize(0, 0) {}

    void setImage(const cv::Mat &frame) override {
        BaseInferRequest::setImage(frame);

        srcImageSize.height = frame.rows;
        srcImageSize.width = frame.cols;
    }

    void setSourceImageSize(int width, int height) {
        srcImageSize.height = height;
        srcImageSize.width = width;
    }

    cv::Size getSourceImageSize() {
        return srcImageSize;
    }

    using Ptr = std::shared_ptr<VehicleDetectionInferRequest>;
};

struct VehicleDetection : BaseDetection {
    int maxProposalCount;
    int objectSize;

    struct Result {
        int label;
        float confidence;
        cv::Rect location;
        size_t channelId;
    };

    VehicleDetection() : BaseDetection(FLAGS_m, "Vehicle"), maxProposalCount(0), objectSize(0) {
    }

    VehicleDetectionInferRequest::Ptr createInferRequest() {
        if (!enabled())
            return nullptr;

        return std::make_shared<VehicleDetectionInferRequest>(net, inputName);
    }

    CNNNetwork read() override {
        slog::info << "Loading network files for VehicleDetection" << slog::endl;
        CNNNetReader netReader;
        /** Read network model **/
        netReader.ReadNetwork(FLAGS_m);
        /** Set batch size to 1 **/
        slog::info << "Batch size is forced to  1" << slog::endl;
        netReader.getNetwork().setBatchSize(1);
        /** Extract model name and load it's weights **/
        std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
        netReader.ReadWeights(binFileName);
        // -----------------------------------------------------------------------------------------------------

        /** SSD-based network should have one input and one output **/
        // ---------------------------Check inputs ------------------------------------------------------
        slog::info << "Checking Vehicle Detection inputs" << slog::endl;
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
        slog::info << "Checking Vehicle Detection outputs" << slog::endl;
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

        slog::info << "Loading Vehicle Detection model to the "<< FLAGS_d << " plugin" << slog::endl;
        return netReader.getNetwork();
    }

    void fetchResults(VehicleDetectionInferRequest::Ptr request, std::vector<Result>& results) {
        cv::Size srcImageSize = request->getSourceImageSize();
        size_t channelId = request->getId();
        const float *detections = request->getBlob(outputName)->buffer().as<float *>();
        // pretty much regular SSD post-processing
        for (int i = 0; i < maxProposalCount; i++) {
            float image_id = detections[i * objectSize + 0];  // in case of batch
            Result r;
            r.label = static_cast<int>(detections[i * objectSize + 1]);
            r.confidence = detections[i * objectSize + 2];
            if (r.confidence <= FLAGS_t) {
                continue;
            }

            r.location.x = static_cast<int>(detections[i * objectSize + 3] * srcImageSize.width);
            r.location.y = static_cast<int>(detections[i * objectSize + 4] * srcImageSize.height);
            r.location.width = static_cast<int>(detections[i * objectSize + 5] * srcImageSize.width - r.location.x);
            r.location.height = static_cast<int>(detections[i * objectSize + 6] * srcImageSize.height - r.location.y);

            r.channelId = channelId;

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

    BaseInferRequest::Ptr createInferRequest() {
        if (!enabled())
            return nullptr;

        return std::make_shared<BaseInferRequest>(net, inputName);
    }

    struct Attributes { std::string type; std::string color;};
    Attributes GetAttributes(BaseInferRequest::Ptr request) {
        static const std::string colors[] = {
                "white", "gray", "yellow", "red", "green", "blue", "black"
        };
        static const std::string types[] = {
                "car", "bus", "truck", "van"
        };

        // 7 possible colors for each vehicle and we should select the one with the maximum probability
        auto colorsValues = request->getBlob(outputNameForColor)->buffer().as<float*>();
        // 4 possible types for each vehicle and we should select the one with the maximum probability
        auto typesValues  = request->getBlob(outputNameForType)->buffer().as<float*>();

        const auto color_id = std::max_element(colorsValues, colorsValues + 7) - colorsValues;
        const auto type_id =  std::max_element(typesValues,  typesValues  + 4) - typesValues;

        return {types[type_id], colors[color_id]};
    }

    CNNNetwork read() override {
        slog::info << "Loading network files for VehicleAttribs" << slog::endl;
        CNNNetReader netReader;
        /** Read network model **/
        netReader.ReadNetwork(FLAGS_m_va);
        netReader.getNetwork().setBatchSize(1);
        slog::info << "Batch size is forced to 1 for Vehicle Attribs" << slog::endl;

        /** Extract model name and load it's weights **/
        std::string binFileName = fileNameNoExt(FLAGS_m_va) + ".bin";
        netReader.ReadWeights(binFileName);
        // -----------------------------------------------------------------------------------------------------

        /** Vehicle Attribs network should have one input two outputs **/
        // ---------------------------Check inputs ------------------------------------------------------
        slog::info << "Checking VehicleAttribs inputs" << slog::endl;
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
        slog::info << "Checking Vehicle Attribs outputs" << slog::endl;
        OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
        if (outputInfo.size() != 2) {
            throw std::logic_error("Vehicle Attribs Network expects networks having two outputs");
        }
        auto it = outputInfo.begin();
        outputNameForColor = (it++)->second->name;  // color is the first output
        outputNameForType = (it++)->second->name;  // type is the second output

        slog::info << "Loading Vehicle Attribs model to the "<< FLAGS_d_va << " plugin" << slog::endl;
        _enabled = true;
        return netReader.getNetwork();
    }
};

struct LPRInferRequest : BaseInferRequest {
    std::string inputSeqName;

    LPRInferRequest(ExecutableNetwork& net, std::string& inputName, std::string& inputSeqName) :
        BaseInferRequest(net, inputName), inputSeqName(inputSeqName) {}

    void fillSeqBlob() {
        Blob::Ptr seqBlob = request->GetBlob(inputSeqName);
        int maxSequenceSizePerPlate = seqBlob->getTensorDesc().getDims()[0];
        // second input is sequence, which is some relic from the training
        // it should have the leading 0.0f and rest 1.0f
        float* blob_data = seqBlob->buffer().as<float*>();
        blob_data[0] = 0.0f;
        std::fill(blob_data + 1, blob_data + maxSequenceSizePerPlate, 1.0f);
    }

    void setBlob(const Blob::Ptr &frameBlob) override {
        BaseInferRequest::setBlob(frameBlob);
        if (!inputSeqName.empty()) {
            fillSeqBlob();
        }
    }

    void setImage(const cv::Mat &frame) override {
        BaseInferRequest::setImage(frame);
        if (!inputSeqName.empty()) {
            fillSeqBlob();
        }
    }

    using Ptr = std::shared_ptr<LPRInferRequest>;
};

struct LPRDetection : BaseDetection {
    std::string inputSeqName;
    const size_t maxSequenceSizePerPlate = 88;

    LPRDetection() : BaseDetection(FLAGS_m_lpr, "License Plate Recognition") {}

    BaseInferRequest::Ptr createInferRequest() {
        if (!enabled())
            return nullptr;

        return std::make_shared<LPRInferRequest>(net, inputName, inputSeqName);
    }

    std::string GetLicencePlateText(BaseInferRequest::Ptr request) {
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
        const auto data = request->getBlob(outputName)->buffer().as<float*>();
        std::string result;
        for (size_t i = 0; i < maxSequenceSizePerPlate; i++) {
            if (data[i] == -1)
                break;
            result += items[static_cast<size_t>(data[i])];
        }

        return result;
    }
    CNNNetwork read() override {
        slog::info << "Loading network files for Licence Plate Recognition (LPR)" << slog::endl;
        CNNNetReader netReader;
        /** Read network model **/
        netReader.ReadNetwork(FLAGS_m_lpr);
        slog::info << "Batch size is forced to  1 for LPR Network" << slog::endl;
        netReader.getNetwork().setBatchSize(1);
        /** Extract model name and load it's weights **/
        std::string binFileName = fileNameNoExt(FLAGS_m_lpr) + ".bin";
        netReader.ReadWeights(binFileName);

        /** LPR network should have 2 inputs (and second is just a stub) and one output **/
        // ---------------------------Check inputs ------------------------------------------------------
        slog::info << "Checking LPR Network inputs" << slog::endl;
        InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
        if (inputInfo.size() == 2) {
            auto sequenceInput = (++inputInfo.begin());
            inputSeqName = sequenceInput->first;
            if (sequenceInput->second->getTensorDesc().getDims()[0] != maxSequenceSizePerPlate) {
                throw std::logic_error("LPR post-processing assumes certain maximum sequences");
            }
        } else if (inputInfo.size() == 1) {
            inputSeqName = "";
        } else {
            throw std::logic_error("LPR should have 1 or 2 inputs");
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
        slog::info << "Checking LPR Network outputs" << slog::endl;
        OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
        if (outputInfo.size() != 1) {
            throw std::logic_error("LPR should have 1 output");
        }
        outputName = outputInfo.begin()->first;
        slog::info << "Loading LPR model to the "<< FLAGS_d_lpr << " plugin" << slog::endl;

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

struct VehicleObject {
    std::string type;
    std::string color;
    cv::Rect location;
    int channelId;
};

struct LicensePlateObject {
    std::string text;
    cv::Rect location;
    int channelId;
};

struct GridMat {
    cv::Mat outimg;
    cv::Size cellSize;
    std::vector<cv::Point> points;
    int width;
    int height;

    explicit GridMat(size_t dispWidth, size_t dispHeight, size_t nChannels, std::vector<cv::VideoCapture>& cap, std::vector<cv::Mat>& images) {
        size_t maxWidth = 0;
        size_t maxHeight = 0;
        for (size_t i = 0; i < cap.size(); i++) {
            maxWidth = std::max(maxWidth, (size_t) cap[i].get(cv::CAP_PROP_FRAME_WIDTH));
            maxHeight = std::max(maxHeight, (size_t) cap[i].get(cv::CAP_PROP_FRAME_HEIGHT));
        }

        for (size_t i = 0; i < images.size(); i++) {
            maxWidth = std::max(maxWidth, (size_t) images[i].cols);
            maxHeight = std::max(maxHeight, (size_t) images[i].rows);
        }

        size_t nGridCols = static_cast<size_t>(ceil(sqrt(static_cast<float>(nChannels))));
        size_t nGridRows = (nChannels - 1) / nGridCols + 1;
        size_t gridMaxWidth = static_cast<size_t>(dispWidth/nGridCols);
        size_t gridMaxHeight = static_cast<size_t>(dispHeight/nGridRows);

        if (maxWidth == 0) {
            throw std::logic_error("Image max width can't be equal to 0");
        }
        if (maxHeight == 0) {
            throw std::logic_error("Image max height can't be equal to 0");
        }
        float scaleWidth = static_cast<float>(gridMaxWidth) / maxWidth;
        float scaleHeight = static_cast<float>(gridMaxHeight) / maxHeight;
        float scaleFactor = std::min(1.f, std::min(scaleWidth, scaleHeight));

        cellSize.width = static_cast<int>(maxWidth * scaleFactor);
        cellSize.height = static_cast<int>(maxHeight * scaleFactor);

        for (size_t i = 0; i < nChannels; i++) {
            cv::Point p;
            p.x = cellSize.width * (i % nGridCols);
            p.y = cellSize.height * (i / nGridCols);
            points.push_back(p);
        }

        height = cellSize.height * nGridRows;
        width = cellSize.width * nGridCols;
        outimg.create(height, width, CV_8UC3);
        outimg.setTo(0);
    }

    void fill(std::vector<cv::Mat>& frames) {
        if (frames.size() > points.size()) {
            throw std::logic_error("Cannot display " + std::to_string(frames.size()) + " channels in a grid with " + std::to_string(points.size()) + " cells");
        }

        for (size_t i = 0; i < frames.size(); i++) {
            cv::Mat cell = outimg(cv::Rect(points[i].x, points[i].y, cellSize.width, cellSize.height));

            if ((cellSize.width == frames[i].cols) && (cellSize.height == frames[i].rows)) {
                frames[i].copyTo(cell);
            } else if ((cellSize.width > frames[i].cols) && (cellSize.height > frames[i].rows)) {
                frames[i].copyTo(cell(cv::Rect(0, 0, frames[i].cols, frames[i].rows)));
            } else {
                cv::resize(frames[i], cell, cellSize);
            }
        }
    }

    const cv::Mat& getMat() {
        return outimg;
    }
};

struct CallStat {
    typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

    CallStat():
        numberOfCalls(0), totalDuration(0.0), lastCallDuration(0.0), smoothedDuration(-1.0) {
    }

    void setCallDuration(ms value) {
        lastCallDuration = value.count();
        numberOfCalls++;
        totalDuration += lastCallDuration;
        if (smoothedDuration < 0) {
            smoothedDuration = lastCallDuration;
        }
        double alpha = 0.1;
        smoothedDuration = smoothedDuration * (1.0 - alpha) + lastCallDuration * alpha;
    }

    void start() {
        lastCallStart = std::chrono::high_resolution_clock::now();
    }

    void finish() {
        auto t = std::chrono::high_resolution_clock::now();
        setCallDuration(std::chrono::duration_cast<ms>(t - lastCallStart));
    }

    double avarageTotalDuration() {
        return totalDuration / numberOfCalls;
    }

    size_t numberOfCalls;
    double totalDuration;
    double lastCallDuration;
    double smoothedDuration;
    std::chrono::time_point<std::chrono::high_resolution_clock> lastCallStart;
};

void fillROIColor(GridMat& displayImage, cv::Rect roi, cv::Scalar color, double opacity) {
    if (opacity > 0) {
        roi = roi & cv::Rect(0, 0, displayImage.width, displayImage.height);
        cv::Mat textROI = displayImage.getMat()(roi);
        cv::addWeighted(color, opacity, textROI, 1.0 - opacity , 0.0, textROI);
    }
}

void putTextOnImage(GridMat& displayImage, std::string str, cv::Point p,
                    cv::HersheyFonts font, double fontScale, cv::Scalar color,
                    int thickness, cv::Scalar bgcolor = cv::Scalar(),
                    double opacity = 0) {
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(str, font, 0.5, 1, &baseline);
    fillROIColor(displayImage, cv::Rect(cv::Point(p.x, p.y + baseline),
                                        cv::Point(p.x + textSize.width, p.y - textSize.height)),
                 bgcolor, opacity);
    cv::putText(displayImage.getMat(), str, p, font, fontScale, color, thickness);
}

/** Map {device : plugin} for each topology **/
std::map<std::string, InferencePlugin> pluginsForNetworks;

int main(int argc, char *argv[]) {
    try {
        /** This demo covers 3 certain topologies and cannot be generalized **/
        slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << slog::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        std::vector<cv::VideoCapture> cap;
        std::vector<cv::Mat> images;
        std::vector<std::string> videoFiles;
        std::map<std::string, CallStat> timers;

        if (FLAGS_i == "cam") {
            slog::info << "Capturing video streams from the cameras" << slog::endl;
            slog::info << "Number of input web cams:    " << FLAGS_nc << slog::endl;

            if (FLAGS_nc > 0) {
                slog::info << "Trying to connect " << FLAGS_nc << " web cams ..." << slog::endl;
                for (size_t i = 0; i < FLAGS_nc; i++) {
                    cv::VideoCapture camera(i);
                    if (!camera.isOpened()) {
                        throw std::logic_error("Cannot open camera: " + std::to_string(i));
                    }

                    cap.push_back(camera);
                }
            }
        } else {
            slog::info << "Capturing video streams from the video files or loading images" << slog::endl;
            /** This vector stores paths to the processed videos and images **/
            std::vector<std::string> inputFiles;
            parseInputFilesArguments(inputFiles);

            for (auto && name : inputFiles) {
                timers["capture"].start();
                cv::Mat frame = cv::imread(name, cv::IMREAD_COLOR);
                if (frame.empty()) {
                    videoFiles.push_back(name);
                } else {
                    images.push_back(frame);
                }
                timers["capture"].finish();
            }

            size_t nImageFiles = images.size();
            size_t nVideoFiles = videoFiles.size();

            slog::info << "Number of input image files: " << nImageFiles << slog::endl;
            slog::info << "Number of input video files: " << nVideoFiles << slog::endl;

            if (nVideoFiles > 0) {
                slog::info << "Trying to open input video ..." << slog::endl;
                for (auto && videoFile : videoFiles) {
                    cv::VideoCapture video(videoFile);
                    if (!video.isOpened()) {
                        throw std::logic_error("Cannot open input file: " + videoFile);
                    }

                    cap.push_back(video);
                }
            }
        }

        size_t nInputChannels = (FLAGS_ni >= 0) ? FLAGS_ni : (cap.size() + images.size());
        slog::info << "Number of input channels: " << nInputChannels << slog::endl;

        size_t dispWidth, dispHeight;
        size_t found = FLAGS_display_resolution.find("x");
        dispWidth = std::stoi(FLAGS_display_resolution.substr(0, found));
        dispHeight = std::stoi(FLAGS_display_resolution.substr(found + 1, FLAGS_display_resolution.length()));
        slog::info << "Display resolution: " << FLAGS_display_resolution << slog::endl;

        /** Create display image **/
        GridMat displayImage(dispWidth, dispHeight, nInputChannels, cap, images);

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Load Plugin for inference engine -------------------------------------
        LPRDetection LPR;
        VehicleDetection VehicleDetector;
        VehicleAttribsDetection VehicleAttribs;

        std::vector<std::string> pluginNames = {
                FLAGS_d,
                VehicleAttribs.enabled() ? FLAGS_d_va : "",
                LPR.enabled() ? FLAGS_d_lpr : ""
        };

        for (auto && flag : pluginNames) {
            if (flag == "") continue;
            auto i = pluginsForNetworks.find(flag);
            if (i != pluginsForNetworks.end()) {
                continue;
            }
            slog::info << "Loading plugin " << flag << slog::endl;
            InferencePlugin plugin = PluginDispatcher().getPluginByDevice(flag);

            /** Printing plugin version **/
            printPluginVersion(plugin, std::cout);

            if ((flag.find("CPU") != std::string::npos)) {
                /** Load default extensions lib for the CPU plugin (e.g. SSD's DetectionOutput)**/
                plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());

                if (!FLAGS_l.empty()) {
                    // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
                    auto extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
                    plugin.AddExtension(extension_ptr);
                    slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
                }

                if (nInputChannels > 1) {
                    plugin.SetConfig({{PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, PluginConfigParams::CPU_THROUGHPUT_AUTO}});
                }
            }

            if ((flag.find("GPU") != std::string::npos) && !FLAGS_c.empty()) {
                // Load any user-specified clDNN Extensions
                plugin.SetConfig({ { PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c } });
            }

            if ((flag.find("FPGA") != std::string::npos) && !FLAGS_fpga_device_ids.empty()) {
                plugin.SetConfig({ { InferenceEngine::PluginConfigParams::KEY_DEVICE_ID, FLAGS_fpga_device_ids } });

                if (FLAGS_fpga_device_ids.find(",") != std::string::npos) {
                    plugin.SetConfig({ { "DLIA_DEVICE_REQUEST_PROCESSING", "DLIA_SERIAL" } });
                }
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
        Load(VehicleDetector).into(pluginsForNetworks[FLAGS_d]);
        Load(VehicleAttribs).into(pluginsForNetworks[FLAGS_d_va]);
        Load(LPR).into(pluginsForNetworks[FLAGS_d_lpr]);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Create inferense requests --------------------------------------------
        std::queue<VehicleDetectionInferRequest::Ptr> availableVDetectionRequests, pendingVDetectionRequests;
        std::queue<BaseInferRequest::Ptr> availableVAttribsRequestes, pendingVAttribsRequestes;
        std::queue<BaseInferRequest::Ptr> availableLPRRequests, pendingLPRRequests;
        for (size_t i = 0; i < FLAGS_nireq; i++) {
            availableVDetectionRequests.push(VehicleDetector.createInferRequest());
            availableVAttribsRequestes.push(VehicleAttribs.createInferRequest());
            availableLPRRequests.push(LPR.createInferRequest());
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Do inference ---------------------------------------------------------
        std::vector<cv::Mat> frames(nInputChannels);  // frame images
        std::vector<Blob::Ptr> frameBlobs(nInputChannels);  // blobs include pixel data from each frame

        slog::info << "Start inference " << slog::endl;

        /** Start inference & calc performance **/
        size_t nImageChannels = std::min(nInputChannels, images.size());
        size_t nVideoChannels = std::min(nInputChannels - nImageChannels, cap.size());
        size_t nChannels = nImageChannels + nVideoChannels;
        size_t nExtChannels = nInputChannels - nChannels;
        bool isVideo = (nImageChannels == 0);
        int pause(isVideo ? 1 :0);
        timers["total"].start();

        if (!FLAGS_no_show) {
            std::cout << "To close the application, press 'CTRL+C' or any key with focus on the output window" << std::endl;
        }

        do {
            timers["capture"].start();
            /** set images **/
            for (size_t i = 0; i < nImageChannels; i++) {
                frames[i] = images[i];
            }

            /** decode frames **/
            bool endOfVideo = false;
            for (size_t i = 0; i < nVideoChannels; i++) {
                if (!cap[i].read(frames[i + nImageChannels])) {
                    if (frames[i + nImageChannels].empty()) {
                         endOfVideo = true;  // end of video file
                         break;
                    }
                    throw std::logic_error("Failed to get frame from cv::VideoCapture from " + std::to_string(i) + " channel");
                }
            }

            /** added extra channels **/
            for (size_t i = 0; i < nExtChannels; i++) {
                frames[i + nChannels] = frames[i % nChannels].clone();
            }

            /** stop processing if one of the input channel is empty **/
            if (endOfVideo) {
                if (FLAGS_loop_video) {
                    slog::info << "Trying to reopen input video ... " << slog::endl;
                    for (size_t i = 0; i < videoFiles.size(); i++) {
                        if (!cap[i].open(videoFiles[i])) {
                            throw std::logic_error("Cannot reopen input file: " + videoFiles[i]);
                        }
                    }
                    continue;
                }
                break;
            }
            timers["capture"].finish();

            /** inference **/
            std::vector<VehicleObject> vehicles;
            std::vector<LicensePlateObject> licensePlates;
            std::vector<VehicleDetection::Result> results;
            timers["inference"].start();
            for (size_t i = 0; i < nInputChannels; i++) {
                if (availableVDetectionRequests.empty()) {
                    // ----------------------------Get vehicle detection results -------------------------------------------
                    VehicleDetectionInferRequest::Ptr vehicleDetectionRequest = pendingVDetectionRequests.front();

                    vehicleDetectionRequest->wait();

                    VehicleDetector.fetchResults(vehicleDetectionRequest, results);

                    timers["vehicle_detector"].setCallDuration(vehicleDetectionRequest->getTime());

                    pendingVDetectionRequests.pop();
                    availableVDetectionRequests.push(vehicleDetectionRequest);
                    // -----------------------------------------------------------------------------------------------------
                }

                // ----------------------------Asynchronous run of a vehicle detection inference -----------------------
                VehicleDetectionInferRequest::Ptr vehicleDetectionRequest = availableVDetectionRequests.front();

                vehicleDetectionRequest->setId(i);

                if (FLAGS_auto_resize) {
                    // just wrap Mat object with Blob::Ptr without additional memory allocation
                    frameBlobs[i] = wrapMat2Blob(frames[i]);
                    vehicleDetectionRequest->setBlob(frameBlobs[i]);
                    vehicleDetectionRequest->setSourceImageSize(frames[i].cols, frames[i].rows);
                } else {
                    vehicleDetectionRequest->setImage(frames[i]);
                }

                vehicleDetectionRequest->startAsync();

                availableVDetectionRequests.pop();
                pendingVDetectionRequests.push(vehicleDetectionRequest);
                // -----------------------------------------------------------------------------------------------------
            }

            size_t pidx = 0;
            while (!pendingVDetectionRequests.empty()) {
                if (pidx == results.size()) {
                    // ----------------------------Get vehicle detection results -------------------------------------------
                    VehicleDetectionInferRequest::Ptr vehicleDetectionRequest = pendingVDetectionRequests.front();

                    vehicleDetectionRequest->wait();

                    VehicleDetector.fetchResults(vehicleDetectionRequest, results);

                    timers["vehicle_detector"].setCallDuration(vehicleDetectionRequest->getTime());

                    pendingVDetectionRequests.pop();
                    availableVDetectionRequests.push(vehicleDetectionRequest);
                    // -----------------------------------------------------------------------------------------------------
                }

                for (; pidx < results.size(); pidx++) {
                    if (results[pidx].label == 1) {  // vehicle
                        if (VehicleAttribs.enabled()) {
                            // ----------------------------Run Vehicle Attributes Classification -----------------------
                            if (availableVAttribsRequestes.empty()) {
                                // ----------------------------Get vehicle attributes --------------------------------------
                                BaseInferRequest::Ptr VehicleAttribsRequest = pendingVAttribsRequestes.front();

                                VehicleAttribsRequest->wait();

                                auto attr = VehicleAttribs.GetAttributes(VehicleAttribsRequest);
                                size_t ridx = VehicleAttribsRequest->getId();

                                VehicleObject v;
                                v.location = results[ridx].location;
                                v.color = attr.color;
                                v.type = attr.type;
                                v.channelId = results[ridx].channelId;

                                vehicles.push_back(v);

                                timers["attribs"].setCallDuration(VehicleAttribsRequest->getTime());

                                pendingVAttribsRequestes.pop();
                                availableVAttribsRequestes.push(VehicleAttribsRequest);
                                // -----------------------------------------------------------------------------------------
                            }

                            // -----------------------Asynchronous run of a vehicle attributes classification --------------
                            BaseInferRequest::Ptr VehicleAttribsRequest = availableVAttribsRequestes.front();

                            VehicleAttribsRequest->setId(pidx);

                            cv::Mat frame = frames[results[pidx].channelId];
                            if (FLAGS_auto_resize) {
                                ROI cropRoi;
                                cropRoi.posX = (results[pidx].location.x < 0) ? 0 : results[pidx].location.x;
                                cropRoi.posY = (results[pidx].location.y < 0) ? 0 : results[pidx].location.y;
                                cropRoi.sizeX = std::min((size_t) results[pidx].location.width, frame.cols - cropRoi.posX);
                                cropRoi.sizeY = std::min((size_t) results[pidx].location.height, frame.rows - cropRoi.posY);
                                Blob::Ptr roiBlob = make_shared_blob(frameBlobs[results[pidx].channelId], cropRoi);
                                VehicleAttribsRequest->setBlob(roiBlob);
                            } else {
                                // To crop ROI manually and allocate required memory (cv::Mat) again
                                auto clippedRect = results[pidx].location & cv::Rect(0, 0, frame.cols, frame.rows);
                                cv::Mat vehicle = frame(clippedRect);
                                VehicleAttribsRequest->setImage(vehicle);
                            }

                            VehicleAttribsRequest->startAsync();

                            availableVAttribsRequestes.pop();
                            pendingVAttribsRequestes.push(VehicleAttribsRequest);
                            // -----------------------------------------------------------------------------------------
                        } else {
                            VehicleObject v;
                            v.location = results[pidx].location;
                            v.channelId = results[pidx].channelId;
                            vehicles.push_back(v);
                        }
                    } else {
                        if (LPR.enabled()) {  // licence plate
                            // ----------------------------Run License Plate Recognition -------------------------------
                            if (availableLPRRequests.empty()) {
                                // ----------------------------Get License Plate Text --------------------------------------
                                BaseInferRequest::Ptr LPRRequest = pendingLPRRequests.front();

                                LPRRequest->wait();

                                std::string text = LPR.GetLicencePlateText(LPRRequest);
                                size_t ridx = LPRRequest->getId();

                                LicensePlateObject lp;
                                lp.location = results[ridx].location;
                                lp.text = text;
                                lp.channelId = results[ridx].channelId;

                                licensePlates.push_back(lp);

                                timers["lpr"].setCallDuration(LPRRequest->getTime());

                                pendingLPRRequests.pop();
                                availableLPRRequests.push(LPRRequest);
                                // -----------------------------------------------------------------------------------------
                            }

                            // ----------------------------Asynchronous run of a license plate decoding ----------------
                            BaseInferRequest::Ptr LPRRequest = availableLPRRequests.front();

                            LPRRequest->setId(pidx);

                            cv::Mat frame = frames[results[pidx].channelId];
                            // expanding a bounding box a bit, better for the license plate recognition
                            results[pidx].location.x -= 5;
                            results[pidx].location.y -= 5;
                            results[pidx].location.width += 10;
                            results[pidx].location.height += 10;
                            if (FLAGS_auto_resize) {
                                ROI cropRoi;
                                cropRoi.posX = (results[pidx].location.x < 0) ? 0 : results[pidx].location.x;
                                cropRoi.posY = (results[pidx].location.y < 0) ? 0 : results[pidx].location.y;
                                cropRoi.sizeX = std::min((size_t) results[pidx].location.width, frame.cols - cropRoi.posX);
                                cropRoi.sizeY = std::min((size_t) results[pidx].location.height, frame.rows - cropRoi.posY);
                                Blob::Ptr roiBlob = make_shared_blob(frameBlobs[results[pidx].channelId], cropRoi);
                                LPRRequest->setBlob(roiBlob);
                            } else {
                                // To crop ROI manually and allocate required memory (cv::Mat) again
                                auto clippedRect = results[pidx].location & cv::Rect(0, 0, frame.cols, frame.rows);
                                cv::Mat plate = frame(clippedRect);
                                LPRRequest->setImage(plate);
                            }

                            LPRRequest->startAsync();

                            availableLPRRequests.pop();
                            pendingLPRRequests.push(LPRRequest);
                            // -----------------------------------------------------------------------------------------
                        } else {
                            LicensePlateObject lp;
                            lp.location = results[pidx].location;
                            lp.channelId = results[pidx].channelId;
                            licensePlates.push_back(lp);
                        }
                    }
                }
            }

            // ----------------------------Get results from pending requests --------------------------------------
            while (!pendingVAttribsRequestes.empty()) {
                // ----------------------------Get vehicle attributes --------------------------------------
                BaseInferRequest::Ptr VehicleAttribsRequest = pendingVAttribsRequestes.front();

                VehicleAttribsRequest->wait();

                auto attr = VehicleAttribs.GetAttributes(VehicleAttribsRequest);
                size_t ridx = VehicleAttribsRequest->getId();

                VehicleObject v;
                v.location = results[ridx].location;
                v.color = attr.color;
                v.type = attr.type;
                v.channelId = results[ridx].channelId;

                vehicles.push_back(v);

                timers["attribs"].setCallDuration(VehicleAttribsRequest->getTime());

                pendingVAttribsRequestes.pop();
                availableVAttribsRequestes.push(VehicleAttribsRequest);
                // -----------------------------------------------------------------------------------------
            }

            while (!pendingLPRRequests.empty()) {
                // ----------------------------Get License Plate Text --------------------------------------
                BaseInferRequest::Ptr LPRRequest = pendingLPRRequests.front();

                LPRRequest->wait();

                std::string text = LPR.GetLicencePlateText(LPRRequest);
                size_t ridx = LPRRequest->getId();

                LicensePlateObject lp;
                lp.location = results[ridx].location;
                lp.text = text;
                lp.channelId = results[ridx].channelId;

                licensePlates.push_back(lp);

                timers["lpr"].setCallDuration(LPRRequest->getTime());

                pendingLPRRequests.pop();
                availableLPRRequests.push(LPRRequest);
                // -----------------------------------------------------------------------------------------
            }

            timers["inference"].finish();

            // ----------------------------Process outputs-----------------------------------------------------
            timers["render"].start();
            int vehicleRectThinkness = 2;
            int licensePlateRectThinkness = 1;
            int fontScale = 1;
            int fontThinkness = 1;
            int vehicleAttribOffset = 25;
            int lprOffset = 50;
            for (auto && v : vehicles) {
                auto findLP = [&](VehicleObject& vehicle) {
                    auto itlp = licensePlates.begin();
                    for (; itlp != licensePlates.end(); itlp++) {
                        if ((itlp->channelId == vehicle.channelId) &&
                           ((itlp->location & vehicle.location).area() > 0)) {
                            return itlp;
                        }
                    }
                    return itlp;
                };

                auto lp = findLP(v);

                if (lp == licensePlates.end()) {
                    continue;
                }

                cv::rectangle(frames[lp->channelId], lp->location, cv::Scalar(0, 255, 0), licensePlateRectThinkness);

                if (!lp->text.empty()) {
                    int y_position = lp->location.y + lp->location.height - lprOffset;
                    if (y_position < 0)
                        y_position = 0;

                    cv::putText(frames[lp->channelId],
                        lp->text,
                        cv::Point2f(static_cast<float>(lp->location.x), static_cast<float>(y_position)),
                        cv::FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale,
                        cv::Scalar(0, 0, 255),
                        fontThinkness);

                    if (FLAGS_r) {
                        std::cout << "License Plate Recognition results:" << lp->text << std::endl;
                    }
                }

                cv::rectangle(frames[v.channelId], v.location, cv::Scalar(0, 255, 0), vehicleRectThinkness);

                if (!v.color.empty() && !v.type.empty()) {
                    cv::putText(frames[v.channelId],
                                v.color,
                                cv::Point2f(static_cast<float>(v.location.x + 5), static_cast<float>(v.location.y + vehicleAttribOffset)),
                                cv::FONT_HERSHEY_COMPLEX_SMALL,
                                fontScale,
                                cv::Scalar(255, 0, 0),
                                fontThinkness);
                    cv::putText(frames[v.channelId],
                                v.type,
                                cv::Point2f(static_cast<float>(v.location.x + 5), static_cast<float>(v.location.y + 2 * vehicleAttribOffset)),
                                cv::FONT_HERSHEY_COMPLEX_SMALL,
                                fontScale,
                                cv::Scalar(255, 0, 0),
                                fontThinkness);
                    if (FLAGS_r) {
                        std::cout << "Vehicle Attributes results:" << v.color << ";" << v.type << std::endl;
                    }
                }
            }

            // ----------------------------Fill display images ------------------------------------------------------
            displayImage.fill(frames);

           // ----------------------------Execution statistics -----------------------------------------------------
            fillROIColor(displayImage, cv::Rect(0, 0, 530, 100), cv::Scalar(255, 0, 0), 0.6);

            std::ostringstream out;
            out << std::fixed << std::setprecision(2)
                << 1000. / (timers["capture"].smoothedDuration +
                            std::max(0., timers["render"].smoothedDuration) +
                            timers["inference"].smoothedDuration) << " fps";
            putTextOnImage(displayImage, out.str(), cv::Point(5, 35), cv::FONT_HERSHEY_TRIPLEX, 1.1,
                           cv::Scalar(255, 255, 255), 2);


            out.str("");
            out << "Inference for " << nInputChannels << ((nInputChannels == 1) ? " stream: " : " streams: ")
                << 1000. / timers["inference"].smoothedDuration << " fps";
            putTextOnImage(displayImage, out.str(), cv::Point(5, 60), cv::FONT_HERSHEY_TRIPLEX, 0.6,
                           cv::Scalar(255, 255, 255), 1);

            out.str("");
            out << "Capture: " << std::fixed << std::setprecision(2)
                << 1000. / timers["capture"].smoothedDuration << " fps";
            if (isVideo) {
                out << "; Render: " << std::max(0., 1000. / timers["render"].smoothedDuration) << " fps";
            }
            putTextOnImage(displayImage, out.str(), cv::Point(5, 85), cv::FONT_HERSHEY_TRIPLEX, 0.6,
                           cv::Scalar(255, 255, 255), 1);

            if (!FLAGS_no_show) {
                cv::imshow("Detection results", displayImage.getMat());
                timers["render"].finish();

                const int key = cv::waitKey(pause);
                if (key == 27) {
                    break;
                } else if (key == 32) {
                    pause = (pause + 1) & 1;
                }
            }
        } while (isVideo);

        timers["total"].finish();

        // -----------------------------------------------------------------------------------------------------
        if (timers["inference"].numberOfCalls > 0) {
            std::cout << std::endl << "Average inference time: " << timers["inference"].avarageTotalDuration() << " ms ("
                      << 1000.f / timers["inference"].avarageTotalDuration() << " fps)" << std::endl;

            if ((nInputChannels == 1) && (timers["vehicle_detector"].numberOfCalls > 0)) {
                std::cout << std::endl << "Average vehicle detection time: " << timers["vehicle_detector"].avarageTotalDuration() << " ms ("
                          << 1000.f / timers["vehicle_detector"].avarageTotalDuration() << " fps)" << std::endl;

                if (timers["attribs"].numberOfCalls > 0) {
                    std::cout << std::endl << "Average vehicle attribs time: " << timers["attribs"].avarageTotalDuration() << " ms ("
                              << 1000.f / timers["attribs"].avarageTotalDuration() << " fps)" << std::endl;
                }
                if (timers["lpr"].numberOfCalls > 0) {
                    std::cout << std::endl << "Average lpr time: " << timers["lpr"].avarageTotalDuration() << " ms ("
                              << 1000.f / timers["lpr"].avarageTotalDuration() << " fps)" << std::endl;
                }
            }
        }

        std::cout << std::endl << "Total execution time: " << timers["total"].totalDuration << std::endl << std::endl;

        /** Show performace results **/
        if (FLAGS_pc) {
            for (auto && plugin : pluginsForNetworks) {
                std::cout << "Performance counts for " << plugin.first << " plugin";
                printPerformanceCountsPlugin(plugin.second, std::cout);
            }
        }

        /** release input channels **/
        for (auto && c : cap) {
            c.release();
        }

        cv::destroyAllWindows();
    }
    catch (const std::exception& error) {
        std::cerr << "[ ERROR ] " << error.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    return 0;
}
