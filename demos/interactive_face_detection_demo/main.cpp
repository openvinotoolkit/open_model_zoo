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
* \brief The entry point for the Inference Engine interactive_face_detection demo application
* \file interactive_face_detection_demo/main.cpp
* \example interactive_face_detection_demo/main.cpp
*/
#include <gflags/gflags.h>
#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <iterator>
#include <map>

#include <inference_engine.hpp>

#include <samples/common.hpp>
#include <samples/slog.hpp>

#include "interactive_face_detection.hpp"
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
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    if (FLAGS_n_ag < 1) {
        throw std::logic_error("Parameter -n_ag cannot be 0");
    }

    if (FLAGS_n_hp < 1) {
        throw std::logic_error("Parameter -n_hp cannot be 0");
    }

    // no need to wait for a key press from a user if an output image/video file is not shown.
    FLAGS_no_wait |= FLAGS_no_show;

    return true;
}

// -------------------------Generic routines for detection networks-------------------------------------------------

struct BaseDetection {
    ExecutableNetwork net;
    InferencePlugin * plugin;
    InferRequest::Ptr request;
    std::string & commandLineFlag;
    std::string topoName;
    const int maxBatch;
    const bool isAsync;

    BaseDetection(std::string &commandLineFlag, std::string topoName, int maxBatch, bool isAsync = false)
        : commandLineFlag(commandLineFlag), topoName(topoName), maxBatch(maxBatch), isAsync(isAsync) {
            if (isAsync) {
                slog::info << "Use async mode for " << topoName << slog::endl;
            }
        }

    virtual ~BaseDetection() {}

    ExecutableNetwork* operator ->() {
        return &net;
    }
    virtual CNNNetwork read()  = 0;

    virtual void submitRequest() {
        if (!enabled() || request == nullptr) return;
        if (isAsync) {
            request->StartAsync();
        } else {
            request->Infer();
        }
    }

    virtual void wait() {
        if (!enabled()|| !request || !isAsync) return;
        request->Wait(IInferRequest::WaitMode::RESULT_READY);
    }
    mutable bool enablingChecked = false;
    mutable bool _enabled = false;

    bool enabled() const  {
        if (!enablingChecked) {
            _enabled = !commandLineFlag.empty();
            if (!_enabled) {
                slog::info << topoName << " DISABLED" << slog::endl;
            }
            enablingChecked = true;
        }
        return _enabled;
    }
    void printPerformanceCounts() {
        if (!enabled()) {
            return;
        }
        slog::info << "Performance counts for " << topoName << slog::endl << slog::endl;
        ::printPerformanceCounts(request->GetPerformanceCounts(), std::cout, false);
    }
};

struct FaceDetectionClass : BaseDetection {
    std::string input;
    std::string output;
    int maxProposalCount;
    int objectSize;
    int enquedFrames = 0;
    float width = 0;
    float height = 0;
    bool resultsFetched = false;
    std::vector<std::string> labels;

    struct Result {
        int label;
        float confidence;
        cv::Rect location;
    };

    std::vector<Result> results;

    void submitRequest() override {
        if (!enquedFrames) return;
        enquedFrames = 0;
        resultsFetched = false;
        results.clear();
        BaseDetection::submitRequest();
    }

    void enqueue(const cv::Mat &frame) {
        if (!enabled()) return;

        if (!request) {
            request = net.CreateInferRequestPtr();
        }

        width = frame.cols;
        height = frame.rows;

        Blob::Ptr  inputBlob = request->GetBlob(input);

        matU8ToBlob<uint8_t>(frame, inputBlob);

        enquedFrames = 1;
    }


    FaceDetectionClass() : BaseDetection(FLAGS_m, "Face Detection", 1, FLAGS_async) {}
    CNNNetwork read() override {
        slog::info << "Loading network files for Face Detection" << slog::endl;
        CNNNetReader netReader;
        /** Read network model **/
        netReader.ReadNetwork(FLAGS_m);
        /** Set batch size to 1 **/
        slog::info << "Batch size is set to " << maxBatch << slog::endl;
        netReader.getNetwork().setBatchSize(maxBatch);
        /** Extract model name and load it's weights **/
        std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
        netReader.ReadWeights(binFileName);
        /** Read labels (if any)**/
        std::string labelFileName = fileNameNoExt(FLAGS_m) + ".labels";

        std::ifstream inputFile(labelFileName);
        std::copy(std::istream_iterator<std::string>(inputFile),
                  std::istream_iterator<std::string>(),
                  std::back_inserter(labels));
        // -----------------------------------------------------------------------------------------------------

        /** SSD-based network should have one input and one output **/
        // ---------------------------Check inputs ------------------------------------------------------
        slog::info << "Checking Face Detection inputs" << slog::endl;
        InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
        if (inputInfo.size() != 1) {
            throw std::logic_error("Face Detection network should have only one input");
        }
        InputInfo::Ptr inputInfoFirst = inputInfo.begin()->second;
        inputInfoFirst->setPrecision(Precision::U8);
        inputInfoFirst->getInputData()->setLayout(Layout::NCHW);
        // -----------------------------------------------------------------------------------------------------

        // ---------------------------Check outputs ------------------------------------------------------
        slog::info << "Checking Face Detection outputs" << slog::endl;
        OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
        if (outputInfo.size() != 1) {
            throw std::logic_error("Face Detection network should have only one output");
        }
        DataPtr& _output = outputInfo.begin()->second;
        output = outputInfo.begin()->first;

        const CNNLayerPtr outputLayer = netReader.getNetwork().getLayerByName(output.c_str());
        if (outputLayer->type != "DetectionOutput") {
            throw std::logic_error("Face Detection network output layer(" + outputLayer->name +
                ") should be DetectionOutput, but was " +  outputLayer->type);
        }

        if (outputLayer->params.find("num_classes") == outputLayer->params.end()) {
            throw std::logic_error("Face Detection network output layer (" +
                output + ") should have num_classes integer attribute");
        }

        const int num_classes = outputLayer->GetParamAsInt("num_classes");
        if (labels.size() != num_classes) {
            if (labels.size() == (num_classes - 1))  // if network assumes default "background" class, having no label
                labels.insert(labels.begin(), "fake");
            else
                labels.clear();
        }
        const SizeVector outputDims = _output->getTensorDesc().getDims();
        maxProposalCount = outputDims[2];
        objectSize = outputDims[3];
        if (objectSize != 7) {
            throw std::logic_error("Face Detection network output layer should have 7 as a last dimension");
        }
        if (outputDims.size() != 4) {
            throw std::logic_error("Face Detection network output dimensions not compatible shoulld be 4, but was " +
                                           std::to_string(outputDims.size()));
        }
        _output->setPrecision(Precision::FP32);
        _output->setLayout(Layout::NCHW);

        slog::info << "Loading Face Detection model to the "<< FLAGS_d << " plugin" << slog::endl;
        input = inputInfo.begin()->first;
        return netReader.getNetwork();
    }

    void fetchResults() {
        if (!enabled()) return;
        results.clear();
        if (resultsFetched) return;
        resultsFetched = true;
        const float *detections = request->GetBlob(output)->buffer().as<float *>();

        for (int i = 0; i < maxProposalCount; i++) {
            float image_id = detections[i * objectSize + 0];
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

            if (image_id < 0) {
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

struct AgeGenderDetection : BaseDetection {
    std::string input;
    std::string outputAge;
    std::string outputGender;
    int enquedFaces = 0;

    AgeGenderDetection() : BaseDetection(FLAGS_m_ag, "Age Gender", FLAGS_n_ag, FLAGS_async) {}

    void submitRequest() override {
        if (!enquedFaces) return;
        if (FLAGS_dyn_ag) {
            request->SetBatch(enquedFaces);
        }
        BaseDetection::submitRequest();
        enquedFaces = 0;
    }

    void enqueue(const cv::Mat &face) {
        if (!enabled()) {
            return;
        }
        if (enquedFaces == maxBatch) {
            slog::warn << "Number of detected faces more than maximum(" << maxBatch << ") processed by Age Gender detector" << slog::endl;
            return;
        }
        if (!request) {
            request = net.CreateInferRequestPtr();
        }

        Blob::Ptr  inputBlob = request->GetBlob(input);

        matU8ToBlob<uint8_t>(face, inputBlob, enquedFaces);

        enquedFaces++;
    }

    struct Result { float age; float maleProb;};
    Result operator[] (int idx) const {
        Blob::Ptr  genderBlob = request->GetBlob(outputGender);
        Blob::Ptr  ageBlob    = request->GetBlob(outputAge);

        return {ageBlob->buffer().as<float*>()[idx] * 100,
                genderBlob->buffer().as<float*>()[idx * 2 + 1]};
    }

    CNNNetwork read() override {
        slog::info << "Loading network files for AgeGender" << slog::endl;
        CNNNetReader netReader;
        // Read network.
        netReader.ReadNetwork(FLAGS_m_ag);

        // Set maximum batch size to be used.
        netReader.getNetwork().setBatchSize(maxBatch);
        slog::info << "Batch size is set to " << netReader.getNetwork().getBatchSize() << " for Age Gender" << slog::endl;


        // Extract model name and load its weights
        std::string binFileName = fileNameNoExt(FLAGS_m_ag) + ".bin";
        netReader.ReadWeights(binFileName);

        // ---------------------------Check inputs ------------------------------------------------------
        // Age Gender network should have one input two outputs
        slog::info << "Checking Age Gender inputs" << slog::endl;
        InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
        if (inputInfo.size() != 1) {
            throw std::logic_error("Age gender topology should have only one input");
        }
        InputInfo::Ptr& inputInfoFirst = inputInfo.begin()->second;
        inputInfoFirst->setPrecision(Precision::U8);
        inputInfoFirst->getInputData()->setLayout(Layout::NCHW);
        input = inputInfo.begin()->first;
        // -----------------------------------------------------------------------------------------------

        // ---------------------------Check outputs ------------------------------------------------------
        slog::info << "Checking Age Gender outputs" << slog::endl;
        OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
        if (outputInfo.size() != 2) {
            throw std::logic_error("Age Gender network should have two output layers");
        }
        auto it = outputInfo.begin();

        DataPtr ptrAgeOutput = (it++)->second;
        DataPtr ptrGenderOutput = (it++)->second;

        if (!ptrAgeOutput) {
            throw std::logic_error("Age output data pointer is not valid");
        }
        if (!ptrGenderOutput) {
            throw std::logic_error("Gender output data pointer is not valid");
        }

        auto genderCreatorLayer = ptrGenderOutput->getCreatorLayer().lock();
        auto ageCreatorLayer = ptrAgeOutput->getCreatorLayer().lock();

        if (!ageCreatorLayer) {
            throw std::logic_error("Age's creator layer pointer is not valid");
        }
        if (!genderCreatorLayer) {
            throw std::logic_error("Gender's creator layer pointer is not valid");
        }

        // if gender output is convolution, it can be swapped with age
        if (genderCreatorLayer->type == "Convolution") {
            std::swap(ptrAgeOutput, ptrGenderOutput);
        }

        if (ptrAgeOutput->getCreatorLayer().lock()->type != "Convolution") {
            throw std::logic_error("In Age Gender network, age layer (" + ageCreatorLayer->name +
                ") should be a Convolution, but was: " + ageCreatorLayer->type);
        }

        if (ptrGenderOutput->getCreatorLayer().lock()->type != "SoftMax") {
            throw std::logic_error("In Age Gender network, gender layer (" + genderCreatorLayer->name +
                ") should be a SoftMax, but was: " + genderCreatorLayer->type);
        }
        slog::info << "Age layer: " << ageCreatorLayer->name<< slog::endl;
        slog::info << "Gender layer: " << genderCreatorLayer->name<< slog::endl;

        outputAge = ptrAgeOutput->name;
        outputGender = ptrGenderOutput->name;

        slog::info << "Loading Age Gender model to the "<< FLAGS_d_ag << " plugin" << slog::endl;
        _enabled = true;
        return netReader.getNetwork();
    }
};

struct HeadPoseDetection : BaseDetection {
    std::string input;
    std::string outputAngleR = "angle_r_fc";
    std::string outputAngleP = "angle_p_fc";
    std::string outputAngleY = "angle_y_fc";
    int enquedFaces = 0;
    cv::Mat cameraMatrix;
    HeadPoseDetection() : BaseDetection(FLAGS_m_hp, "Head Pose", FLAGS_n_hp, FLAGS_async) {}

    void submitRequest() override {
        if (!enquedFaces) return;
        if (FLAGS_dyn_hp) {
            request->SetBatch(enquedFaces);
        }
        BaseDetection::submitRequest();
        enquedFaces = 0;
    }

    void enqueue(const cv::Mat &face) {
        if (!enabled()) {
            return;
        }
        if (enquedFaces == maxBatch) {
            slog::warn << "Number of detected faces more than maximum(" << maxBatch << ") processed by Head Pose detector" << slog::endl;
            return;
        }
        if (!request) {
            request = net.CreateInferRequestPtr();
        }

        Blob::Ptr  inputBlob = request->GetBlob(input);

        matU8ToBlob<uint8_t>(face, inputBlob, enquedFaces);

        enquedFaces++;
    }

    struct Results {
        float angle_r;
        float angle_p;
        float angle_y;
    };

    Results operator[] (int idx) const {
        Blob::Ptr  angleR = request->GetBlob(outputAngleR);
        Blob::Ptr  angleP = request->GetBlob(outputAngleP);
        Blob::Ptr  angleY = request->GetBlob(outputAngleY);

        return {angleR->buffer().as<float*>()[idx],
                angleP->buffer().as<float*>()[idx],
                angleY->buffer().as<float*>()[idx]};
    }

    CNNNetwork read() override {
        slog::info << "Loading network files for Head Pose detection " << slog::endl;
        CNNNetReader netReader;
        // Read network model.
        netReader.ReadNetwork(FLAGS_m_hp);
        // Set maximum batch size.
        netReader.getNetwork().setBatchSize(maxBatch);
        slog::info << "Batch size is set to  " << netReader.getNetwork().getBatchSize() << " for Head Pose Network" << slog::endl;
        // Extract model name and load its weights.
        std::string binFileName = fileNameNoExt(FLAGS_m_hp) + ".bin";
        netReader.ReadWeights(binFileName);

        // ---------------------------Check inputs ------------------------------------------------------
        slog::info << "Checking Head Pose Network inputs" << slog::endl;
        InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
        if (inputInfo.size() != 1) {
            throw std::logic_error("Head Pose topology should have only one input");
        }
        InputInfo::Ptr& inputInfoFirst = inputInfo.begin()->second;
        inputInfoFirst->setPrecision(Precision::U8);
        inputInfoFirst->getInputData()->setLayout(Layout::NCHW);
        input = inputInfo.begin()->first;
        // -----------------------------------------------------------------------------------------------------

        // ---------------------------Check outputs ------------------------------------------------------
        slog::info << "Checking Head Pose network outputs" << slog::endl;
        OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
        if (outputInfo.size() != 3) {
            throw std::logic_error("Head Pose network should have 3 outputs");
        }
        for (auto& output : outputInfo) {
            output.second->setPrecision(Precision::FP32);
            output.second->setLayout(Layout::NC);
        }
        std::map<std::string, bool> layerNames = {
            {outputAngleR, false},
            {outputAngleP, false},
            {outputAngleY, false}
        };

        for (auto && output : outputInfo) {
            CNNLayerPtr layer = output.second->getCreatorLayer().lock();
            if (!layer) {
                throw std::logic_error("Layer pointer is invalid");
            }
            if (layerNames.find(layer->name) == layerNames.end()) {
                throw std::logic_error("Head Pose network output layer unknown: " + layer->name + ", should be " +
                    outputAngleR + " or " + outputAngleP + " or " + outputAngleY);
            }
            if (layer->type != "FullyConnected") {
                throw std::logic_error("Head Pose network output layer (" + layer->name + ") has invalid type: " +
                    layer->type + ", should be FullyConnected");
            }
            auto fc = dynamic_cast<FullyConnectedLayer*>(layer.get());
            if (!fc) {
                throw std::logic_error("Fully connected layer is not valid");
            }
            if (fc->_out_num != 1) {
                throw std::logic_error("Head Pose network output layer (" + layer->name + ") has invalid out-size=" +
                    std::to_string(fc->_out_num) + ", should be 1");
            }
            layerNames[layer->name] = true;
        }

        slog::info << "Loading Head Pose model to the "<< FLAGS_d_hp << " plugin" << slog::endl;

        _enabled = true;
        return netReader.getNetwork();
    }

    void buildCameraMatrix(int cx, int cy, float focalLength) {
        if (!cameraMatrix.empty()) return;
        cameraMatrix = cv::Mat::zeros(3, 3, CV_32F);
        cameraMatrix.at<float>(0) = focalLength;
        cameraMatrix.at<float>(2) = static_cast<float>(cx);
        cameraMatrix.at<float>(4) = focalLength;
        cameraMatrix.at<float>(5) = static_cast<float>(cy);
        cameraMatrix.at<float>(8) = 1;
    }

    void drawAxes(cv::Mat& frame, cv::Point3f cpoint, Results headPose, float scale) {
        double yaw   = headPose.angle_y;
        double pitch = headPose.angle_p;
        double roll  = headPose.angle_r;

        pitch *= CV_PI / 180.0;
        yaw   *= CV_PI / 180.0;
        roll  *= CV_PI / 180.0;

        cv::Matx33f        Rx(1,           0,            0,
                              0,  cos(pitch),  -sin(pitch),
                              0,  sin(pitch),  cos(pitch));
        cv::Matx33f Ry(cos(yaw),           0,    -sin(yaw),
                              0,           1,            0,
                       sin(yaw),           0,    cos(yaw));
        cv::Matx33f Rz(cos(roll), -sin(roll),            0,
                       sin(roll),  cos(roll),            0,
                              0,           0,            1);


        auto r = cv::Mat(Rz*Ry*Rx);
        buildCameraMatrix(frame.cols / 2, frame.rows / 2, 950.0);

        cv::Mat xAxis(3, 1, CV_32F), yAxis(3, 1, CV_32F), zAxis(3, 1, CV_32F), zAxis1(3, 1, CV_32F);

        xAxis.at<float>(0) = 1 * scale;
        xAxis.at<float>(1) = 0;
        xAxis.at<float>(2) = 0;

        yAxis.at<float>(0) = 0;
        yAxis.at<float>(1) = -1 * scale;
        yAxis.at<float>(2) = 0;

        zAxis.at<float>(0) = 0;
        zAxis.at<float>(1) = 0;
        zAxis.at<float>(2) = -1 * scale;

        zAxis1.at<float>(0) = 0;
        zAxis1.at<float>(1) = 0;
        zAxis1.at<float>(2) = 1 * scale;

        cv::Mat o(3, 1, CV_32F, cv::Scalar(0));
        o.at<float>(2) = cameraMatrix.at<float>(0);

        xAxis = r * xAxis + o;
        yAxis = r * yAxis + o;
        zAxis = r * zAxis + o;
        zAxis1 = r * zAxis1 + o;

        cv::Point p1, p2;

        p2.x = static_cast<int>((xAxis.at<float>(0) / xAxis.at<float>(2) * cameraMatrix.at<float>(0)) + cpoint.x);
        p2.y = static_cast<int>((xAxis.at<float>(1) / xAxis.at<float>(2) * cameraMatrix.at<float>(4)) + cpoint.y);
        cv::line(frame, cv::Point(cpoint.x, cpoint.y), p2, cv::Scalar(0, 0, 255), 2);

        p2.x = static_cast<int>((yAxis.at<float>(0) / yAxis.at<float>(2) * cameraMatrix.at<float>(0)) + cpoint.x);
        p2.y = static_cast<int>((yAxis.at<float>(1) / yAxis.at<float>(2) * cameraMatrix.at<float>(4)) + cpoint.y);
        cv::line(frame, cv::Point(cpoint.x, cpoint.y), p2, cv::Scalar(0, 255, 0), 2);

        p1.x = static_cast<int>((zAxis1.at<float>(0) / zAxis1.at<float>(2) * cameraMatrix.at<float>(0)) + cpoint.x);
        p1.y = static_cast<int>((zAxis1.at<float>(1) / zAxis1.at<float>(2) * cameraMatrix.at<float>(4)) + cpoint.y);

        p2.x = static_cast<int>((zAxis.at<float>(0) / zAxis.at<float>(2) * cameraMatrix.at<float>(0)) + cpoint.x);
        p2.y = static_cast<int>((zAxis.at<float>(1) / zAxis.at<float>(2) * cameraMatrix.at<float>(4)) + cpoint.y);
        cv::line(frame, p1, p2, cv::Scalar(255, 0, 0), 2);
        cv::circle(frame, p2, 3, cv::Scalar(255, 0, 0), 2);
    }
};

struct EmotionsDetectionClass : BaseDetection {
    std::string input;
    std::string outputEmotions;
    int enquedFaces = 0;

    EmotionsDetectionClass() : BaseDetection(FLAGS_m_em, "Emotions Recognition", FLAGS_n_em, FLAGS_async) {}

    void submitRequest() override {
        if (!enquedFaces) return;
        if (FLAGS_dyn_em) {
            request->SetBatch(enquedFaces);
        }
        BaseDetection::submitRequest();
        enquedFaces = 0;
    }

    void enqueue(const cv::Mat &face) {
        if (!enabled()) {
            return;
        }
        if (enquedFaces == maxBatch) {
            slog::warn << "Number of detected faces more than maximum(" << maxBatch << ") processed by Emotions detector" << slog::endl;
            return;
        }
        if (!request) {
            request = net.CreateInferRequestPtr();
        }

        Blob::Ptr inputBlob = request->GetBlob(input);

        matU8ToBlob<uint8_t>(face, inputBlob, enquedFaces);

        enquedFaces++;
    }

    std::string operator[] (int idx) const {
        // Vector of supported emotions.
        static const std::vector<std::string> emotionsVec = {"neutral", "happy", "sad", "surprise", "anger"};
        auto emotionsVecSize = emotionsVec.size();

        Blob::Ptr emotionsBlob = request->GetBlob(outputEmotions);

        /* emotions vector must have the same size as number of channels
         * in model output. Default output format is NCHW so we check index 1. */
        int numOfChannels = emotionsBlob->getTensorDesc().getDims().at(1);
        if (numOfChannels != emotionsVec.size()) {
            throw std::logic_error("Output size (" + std::to_string(numOfChannels) +
                                   ") of the Emotions Recognition network is not equal "
                                   "to used emotions vector size (" +
                                   std::to_string(emotionsVec.size()) + ")");
        }

        auto emotionsValues = emotionsBlob->buffer().as<float *>();
        auto outputIdxPos = emotionsValues + idx;

        /* we identify an index of the most probable emotion in output array
           for idx image to return appropriate emotion name */
        int maxProbEmotionIx = std::max_element(outputIdxPos, outputIdxPos + emotionsVecSize) - outputIdxPos;
        return emotionsVec[maxProbEmotionIx];
    }

    CNNNetwork read() override {
        slog::info << "Loading network files for Emotions recognition" << slog::endl;
        InferenceEngine::CNNNetReader netReader;
        // Read network model.
        netReader.ReadNetwork(FLAGS_m_em);

        // Set maximum batch size.
        netReader.getNetwork().setBatchSize(maxBatch);
        slog::info << "Batch size is set to " << netReader.getNetwork().getBatchSize() << " for Emotions recognition" << slog::endl;


        // Extract model name and load its weights.
        std::string binFileName = fileNameNoExt(FLAGS_m_em) + ".bin";
        netReader.ReadWeights(binFileName);

        // ----------------------------------------------------------------------------------------------

        // Emotions recognition network should have one input and one output.
        // ---------------------------Check inputs ------------------------------------------------------
        slog::info << "Checking Emotions Recognition inputs" << slog::endl;
        InferenceEngine::InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
        if (inputInfo.size() != 1) {
            throw std::logic_error("Emotions Recognition topology should have only one input");
        }
        auto& inputInfoFirst = inputInfo.begin()->second;
        inputInfoFirst->setPrecision(Precision::U8);
        inputInfoFirst->getInputData()->setLayout(Layout::NCHW);
        input = inputInfo.begin()->first;
        // -----------------------------------------------------------------------------------------------

        // ---------------------------Check outputs ------------------------------------------------------
        slog::info << "Checking Emotions Recognition outputs" << slog::endl;
        InferenceEngine::OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
        if (outputInfo.size() != 1) {
            throw std::logic_error("Emotions Recognition network should have one output layer");
        }
        for (auto& output : outputInfo) {
            output.second->setPrecision(Precision::FP32);
            output.second->setLayout(Layout::NCHW);
        }

        DataPtr emotionsOutput = outputInfo.begin()->second;

        if (!emotionsOutput) {
            throw std::logic_error("Emotions output data pointer is invalid");
        }

        auto emotionsCreatorLayer = emotionsOutput->getCreatorLayer().lock();

        if (!emotionsCreatorLayer) {
            throw std::logic_error("Emotions creator layer pointer is invalid");
        }

        if (emotionsCreatorLayer->type != "SoftMax") {
            throw std::logic_error("In Emotions Recognition network, Emotion layer ("
                                   + emotionsCreatorLayer->name +
                                   ") should be a SoftMax, but was: " +
                                           emotionsCreatorLayer->type);
        }
        slog::info << "Emotions layer: " << emotionsCreatorLayer->name<< slog::endl;

        outputEmotions = emotionsOutput->name;

        slog::info << "Loading Emotions Recognition model to the "<< FLAGS_d_em << " plugin" << slog::endl;
        _enabled = true;
        return netReader.getNetwork();
    }
};

struct Load {
    BaseDetection& detector;
    explicit Load(BaseDetection& detector) : detector(detector) { }

    void into(InferencePlugin & plg, bool enable_dynamic_batch = false) const {
        if (detector.enabled()) {
            std::map<std::string, std::string> config;
            if (enable_dynamic_batch) {
                config[PluginConfigParams::KEY_DYN_BATCH_ENABLED] = PluginConfigParams::YES;
            }
            detector.net = plg.LoadNetwork(detector.read(), config);
            detector.plugin = &plg;
        }
    }
};


struct CallStat {
    public:
    typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

    double getSmoothedDuration() {
        // Additional check is needed for the first frame while duration of the first
        // visualisation is not calculated yet.
        if (_smoothed_duration < 0) {
            auto t = std::chrono::high_resolution_clock::now();
            return std::chrono::duration_cast<ms>(t - _last_call_start).count();
        }
        return _smoothed_duration;
    }

    double getTotalDuration() {
        return _total_duration;
    }

    void calculateDuration() {
        auto t = std::chrono::high_resolution_clock::now();
        _last_call_duration = std::chrono::duration_cast<ms>(t - _last_call_start).count();
        _number_of_calls++;
        _total_duration += _last_call_duration;
        if (_smoothed_duration < 0) {
            _smoothed_duration = _last_call_duration;
        }
        double alpha = 0.1;
        _smoothed_duration = _smoothed_duration * (1.0 - alpha) + _last_call_duration * alpha;
    }

    void setStartTime() {
        _last_call_start = std::chrono::high_resolution_clock::now();
    }

    private:
    size_t _number_of_calls {0};
    double _total_duration {0.0};
    double _last_call_duration {0.0};
    double _smoothed_duration {-1.0};
    std::chrono::time_point<std::chrono::high_resolution_clock> _last_call_start;
};

class Timer {
    public:
    void start(const std::string& name) {
        if (_timers.find(name) == _timers.end()) {
            _timers[name] = CallStat();
        }
        _timers[name].setStartTime();
    }

    void finish(const std::string& name) {
        auto& timer = (*this)[name];
        timer.calculateDuration();
    }

    CallStat& operator[](const std::string& name) {
        if (_timers.find(name) == _timers.end()) {
            throw std::logic_error("No timer with name " + name + ".");
        }
        return _timers[name];
    }

    private:
    std::map<std::string, CallStat> _timers;
};

int main(int argc, char *argv[]) {
    try {
        std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        slog::info << "Reading input" << slog::endl;
        cv::VideoCapture cap;
        const bool isCamera = FLAGS_i == "cam";
        if (!(FLAGS_i == "cam" ? cap.open(0) : cap.open(FLAGS_i))) {
            throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
        }
        const size_t width  = (size_t) cap.get(CV_CAP_PROP_FRAME_WIDTH);
        const size_t height = (size_t) cap.get(CV_CAP_PROP_FRAME_HEIGHT);

        // read input (video) frame
        cv::Mat frame;
        if (!cap.read(frame)) {
            throw std::logic_error("Failed to get frame from cv::VideoCapture");
        }
        // -----------------------------------------------------------------------------------------------------
        // --------------------------- 1. Load Plugin for inference engine -------------------------------------
        std::map<std::string, InferencePlugin> pluginsForDevices;
        std::vector<std::pair<std::string, std::string>> cmdOptions = {
            {FLAGS_d, FLAGS_m}, {FLAGS_d_ag, FLAGS_m_ag}, {FLAGS_d_hp, FLAGS_m_hp},
            {FLAGS_d_em, FLAGS_m_em}
        };

        FaceDetectionClass FaceDetection;
        AgeGenderDetection AgeGender;
        HeadPoseDetection HeadPose;
        EmotionsDetectionClass EmotionsDetection;

        for (auto && option : cmdOptions) {
            auto deviceName = option.first;
            auto networkName = option.second;

            if (deviceName == "" || networkName == "") {
                continue;
            }

            if (pluginsForDevices.find(deviceName) != pluginsForDevices.end()) {
                continue;
            }
            slog::info << "Loading plugin " << deviceName << slog::endl;
            InferencePlugin plugin = PluginDispatcher({"../../../lib/intel64", ""}).getPluginByDevice(deviceName);

            /** Printing plugin version **/
            printPluginVersion(plugin, std::cout);

            /** Load extensions for the CPU plugin **/
            if ((deviceName.find("CPU") != std::string::npos)) {
                plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());

                if (!FLAGS_l.empty()) {
                    // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
                    auto extension_ptr = make_so_pointer<MKLDNNPlugin::IMKLDNNExtension>(FLAGS_l);
                    plugin.AddExtension(std::static_pointer_cast<IExtension>(extension_ptr));
                }
            } else if (!FLAGS_c.empty()) {
                // Load Extensions for other plugins not CPU
                plugin.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}});
            }
            pluginsForDevices[deviceName] = plugin;
        }

        /** Per layer metrics **/
        if (FLAGS_pc) {
            for (auto && plugin : pluginsForDevices) {
                plugin.second.SetConfig({{PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES}});
            }
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 2. Read IR models and load them to plugins ------------------------------
        // Disable dynamic batching for face detector as long it processes one image at a time.
        Load(FaceDetection).into(pluginsForDevices[FLAGS_d], false);
        Load(AgeGender).into(pluginsForDevices[FLAGS_d_ag], FLAGS_dyn_ag);
        Load(HeadPose).into(pluginsForDevices[FLAGS_d_hp], FLAGS_dyn_hp);
        Load(EmotionsDetection).into(pluginsForDevices[FLAGS_d_em], FLAGS_dyn_em);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Do inference ---------------------------------------------------------
        // Start inference & calc performance.
        slog::info << "Start inference " << slog::endl;
        if (!FLAGS_no_show) {
            std::cout << "Press any key to stop" << std::endl;
        }

        Timer timer;
        timer.start("total");

        std::ostringstream out;
        size_t framesCounter = 0;
        bool frameReadStatus;
        bool isLastFrame;
        cv::Mat prev_frame, next_frame;

        // Detect all faces on the first frame and read the next one.
        timer.start("detection");
        FaceDetection.enqueue(frame);
        FaceDetection.submitRequest();
        timer.finish("detection");

        prev_frame = frame.clone();

        // Read next frame.
        timer.start("video frame decoding");
        frameReadStatus = cap.read(frame);
        timer.finish("video frame decoding");

        while (true) {
            framesCounter++;
            isLastFrame = !frameReadStatus;

            timer.start("detection");
            // Retrieve face detection results for previous frame.
            FaceDetection.wait();
            FaceDetection.fetchResults();
            auto prev_detection_results = FaceDetection.results;

            // No valid frame to infer if previous frame is last.
            if (!isLastFrame) {
                FaceDetection.enqueue(frame);
                FaceDetection.submitRequest();
            }
            timer.finish("detection");

            timer.start("data preprocessing");
            // Fill inputs of face analytics networks.
            for (auto &&face : prev_detection_results) {
                if (AgeGender.enabled() || HeadPose.enabled() || EmotionsDetection.enabled()) {
                    auto clippedRect = face.location & cv::Rect(0, 0, width, height);
                    cv::Mat face = prev_frame(clippedRect);
                    AgeGender.enqueue(face);
                    HeadPose.enqueue(face);
                    EmotionsDetection.enqueue(face);
                }
            }
            timer.finish("data preprocessing");

            // Run age-gender recognition, head pose estimation and emotions recognition simultaneously.
            timer.start("face analytics call");
            if (AgeGender.enabled() || HeadPose.enabled() || EmotionsDetection.enabled()) {
                AgeGender.submitRequest();
                HeadPose.submitRequest();
                EmotionsDetection.submitRequest();
            }
            timer.finish("face analytics call");

            // Read next frame if current one is not last.
            if (!isLastFrame) {
                timer.start("video frame decoding");
                frameReadStatus = cap.read(next_frame);
                timer.finish("video frame decoding");
            }

            timer.start("face analytics wait");
            if (AgeGender.enabled() || HeadPose.enabled() || EmotionsDetection.enabled()) {
                AgeGender.wait();
                HeadPose.wait();
                EmotionsDetection.wait();
            }
            timer.finish("face analytics wait");

            // Visualize results.
            if (!FLAGS_no_show) {
                timer.start("visualization");
                out.str("");
                out << "OpenCV cap/render time: " << std::fixed << std::setprecision(2)
                    << (timer["video frame decoding"].getSmoothedDuration() +
                        timer["visualization"].getSmoothedDuration())
                    << " ms";
                cv::putText(prev_frame, out.str(), cv::Point2f(0, 25), cv::FONT_HERSHEY_TRIPLEX, 0.5,
                            cv::Scalar(255, 0, 0));

                out.str("");
                out << "Face detection time: " << std::fixed << std::setprecision(2)
                    << timer["detection"].getSmoothedDuration()
                    << " ms ("
                    << 1000.f /
                       (timer["detection"].getSmoothedDuration())
                    << " fps)";
                cv::putText(prev_frame, out.str(), cv::Point2f(0, 45), cv::FONT_HERSHEY_TRIPLEX, 0.5,
                            cv::Scalar(255, 0, 0));

                if (HeadPose.enabled() || AgeGender.enabled() || EmotionsDetection.enabled()) {
                    out.str("");
                    out << (AgeGender.enabled() ? "Age Gender " : "")
                        << (AgeGender.enabled() && (HeadPose.enabled() || EmotionsDetection.enabled()) ? "+ " : "")
                        << (HeadPose.enabled() ? "Head Pose " : "")
                        << (HeadPose.enabled() && EmotionsDetection.enabled() ? "+ " : "")
                        << (EmotionsDetection.enabled() ? "Emotions Recognition " : "")
                        << "time: " << std::fixed << std::setprecision(2)
                        << timer["face analytics call"].getSmoothedDuration() +
                           timer["face analytics wait"].getSmoothedDuration()
                        << " ms ";
                    if (!prev_detection_results.empty()) {
                        out << "("
                            << 1000.f / (timer["face analytics call"].getSmoothedDuration() +
                                         timer["face analytics wait"].getSmoothedDuration())
                            << " fps)";
                    }
                    cv::putText(prev_frame, out.str(), cv::Point2f(0, 65), cv::FONT_HERSHEY_TRIPLEX, 0.5,
                                cv::Scalar(255, 0, 0));
                }

                // For every detected face.
                int i = 0;
                for (auto &result : prev_detection_results) {
                    cv::Rect rect = result.location;

                    out.str("");

                    if (AgeGender.enabled() && i < AgeGender.maxBatch) {
                        out << (AgeGender[i].maleProb > 0.5 ? "M" : "F");
                        out << std::fixed << std::setprecision(0) << "," << AgeGender[i].age;
                        if (FLAGS_r) {
                            std::cout << "Predicted gender, age = " << out.str() << std::endl;
                        }
                    } else {
                        out << (result.label < FaceDetection.labels.size() ? FaceDetection.labels[result.label] :
                                std::string("label #") + std::to_string(result.label))
                            << ": " << std::fixed << std::setprecision(3) << result.confidence;
                    }

                    if (EmotionsDetection.enabled() && i < EmotionsDetection.maxBatch) {
                        std::string emotion = EmotionsDetection[i];
                        if (FLAGS_r) {
                            std::cout << "Predicted emotion = " << emotion << std::endl;
                        }
                        out << "," << emotion;
                    }

                    cv::putText(prev_frame,
                                out.str(),
                                cv::Point2f(result.location.x, result.location.y - 15),
                                cv::FONT_HERSHEY_COMPLEX_SMALL,
                                0.8,
                                cv::Scalar(0, 0, 255));

                    if (HeadPose.enabled() && i < HeadPose.maxBatch) {
                        if (FLAGS_r) {
                            std::cout << "Head pose results: yaw, pitch, roll = "
                                      << HeadPose[i].angle_y << ";"
                                      << HeadPose[i].angle_p << ";"
                                      << HeadPose[i].angle_r << std::endl;
                        }
                        cv::Point3f center(rect.x + rect.width / 2, rect.y + rect.height / 2, 0);
                        HeadPose.drawAxes(prev_frame, center, HeadPose[i], 50);
                    }
                    auto genderColor = (AgeGender.enabled() && (i < AgeGender.maxBatch)) ?
                                       ((AgeGender[i].maleProb < 0.5) ? cv::Scalar(147, 20, 255) : cv::Scalar(255, 0,
                                                                                                              0))
                                                                                         : cv::Scalar(100, 100, 100);
                    cv::rectangle(prev_frame, result.location, genderColor, 1);
                    i++;
                }

                cv::imshow("Detection results", prev_frame);
                timer.finish("visualization");
            } else if (FLAGS_r) {
                // For every detected face.
                for (int i = 0; i < prev_detection_results.size(); i++) {
                    if (AgeGender.enabled() && i < AgeGender.maxBatch) {
                        out.str("");
                        out << (AgeGender[i].maleProb > 0.5 ? "M" : "F");
                        out << std::fixed << std::setprecision(0) << "," << AgeGender[i].age;
                        std::cout << "Predicted gender, age = " << out.str() << std::endl;
                    }

                    if (EmotionsDetection.enabled() && i < EmotionsDetection.maxBatch) {
                        std::cout << "Predicted emotion = " << EmotionsDetection[i] << std::endl;
                    }

                    if (HeadPose.enabled() && i < HeadPose.maxBatch) {
                        std::cout << "Head pose results: yaw, pitch, roll = "
                                  << HeadPose[i].angle_y << ";"
                                  << HeadPose[i].angle_p << ";"
                                  << HeadPose[i].angle_r << std::endl;
                    }
                }
            }

            // End of file (or a single frame file like an image). We just keep last frame displayed to let user check what was shown
            if (isLastFrame) {
                timer.finish("total");
                if (!FLAGS_no_wait) {
                    std::cout << "No more frames to process. Press any key to exit" << std::endl;
                    cv::waitKey(0);
                }
                break;
            } else if (!FLAGS_no_show && -1 != cv::waitKey(1)) {
                timer.finish("total");
                break;
            }

            prev_frame = frame;
            frame = next_frame;
            next_frame = cv::Mat();
        }

        slog::info << "Number of processed frames: " << framesCounter << slog::endl;
        slog::info << "Total image throughput: " << framesCounter * (1000.f / timer["total"].getTotalDuration()) << " fps" << slog::endl;

        // Show performace results.
        if (FLAGS_pc) {
            FaceDetection.printPerformanceCounts();
            AgeGender.printPerformanceCounts();
            HeadPose.printPerformanceCounts();
            EmotionsDetection.printPerformanceCounts();
        }
        // -----------------------------------------------------------------------------------------------------
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
    return 0;
}
