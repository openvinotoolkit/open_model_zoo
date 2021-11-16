// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>
#include <map>
#include <string>
#include <utility>
#include <vector>

//#include <inference_engine.hpp>
#include "openvino/openvino.hpp"

#include "utils/common.hpp"
#include "utils/ocv_common.hpp"

using namespace ov::preprocess;

class Detector {
public:
    struct Result {
        std::size_t label;
        float confidence;
        cv::Rect location;
    };

    static constexpr int maxProposalCount = 200;
    static constexpr int objectSize = 7;  // Output should have 7 as a last dimension"

    Detector() = default;
//    Detector(InferenceEngine::Core& ie, const std::string& deviceName, const std::string& xmlPath, const std::vector<float>& detectionTresholds,
    Detector(ov::runtime::Core& core, const std::string& deviceName, const std::string& xmlPath, const std::vector<float>& detectionTresholds,
            const bool autoResize, const std::map<std::string, std::string>& pluginConfig) :
        detectionTresholds{detectionTresholds}, m_core{core} {
//        auto network = ie.ReadNetwork(xmlPath);
        auto network = core.read_model(xmlPath);

//        InferenceEngine::InputsDataMap inputInfo(network.getInputsInfo());
//        if (inputInfo.size() != 1) {
//            throw std::logic_error("Detector should have only one input");
//        }
//        InferenceEngine::InputInfo::Ptr& inputInfoFirst = inputInfo.begin()->second;
//        inputInfoFirst->setPrecision(InferenceEngine::Precision::U8);
//        if (autoResize) {
//            inputInfoFirst->getPreProcess().setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
//            inputInfoFirst->setLayout(InferenceEngine::Layout::NHWC);
//        } else {
//            inputInfoFirst->setLayout(InferenceEngine::Layout::NCHW);
//        }

//        detectorInputBlobName = inputInfo.begin()->first;

        ov::OutputVector inputs = network->inputs();
        if (inputs.size() != 1) {
            throw std::logic_error("Detector should have only one input");
        }

        detectorInputBlobName = network->input().get_any_name();

        // ---------------------------Check outputs ------------------------------------------------------
//        InferenceEngine::OutputsDataMap outputInfo(network.getOutputsInfo());
//        if (outputInfo.size() != 1) {
//            throw std::logic_error("Vehicle Detection network should have only one output");
//        }
//        InferenceEngine::DataPtr& _output = outputInfo.begin()->second;
//        const InferenceEngine::SizeVector outputDims = _output->getTensorDesc().getDims();
//        detectorOutputBlobName = outputInfo.begin()->first;
//        if (maxProposalCount != outputDims[2]) {
//            throw std::logic_error("unexpected ProposalCount");
//        }
//        if (objectSize != outputDims[3]) {
//            throw std::logic_error("Output should have 7 as a last dimension");
//        }
//        if (outputDims.size() != 4) {
//            throw std::logic_error("Incorrect output dimensions for SSD");
//        }
//        _output->setPrecision(InferenceEngine::Precision::FP32);

        ov::OutputVector outputs = network->outputs();
        if (outputs.size() != 1) {
            throw std::logic_error("Vehicle Detection network should have only one output");
        }

        ov::Output<ov::Node> output = outputs[0];

        detectorOutputBlobName = output.get_any_name();
        ov::Shape output_shape = output.get_shape();

        if (output_shape.size() != 4) {
            throw std::logic_error("Incorrect output dimensions for SSD");
        }

        if (maxProposalCount != output_shape[2]) {
            throw std::logic_error("unexpected ProposalCount");
        }
        if (objectSize != output_shape[3]) {
            throw std::logic_error("Output should have 7 as a last dimension");
        }

        InputTensorInfo inputTensorInfo;
        inputTensorInfo.set_element_type(ov::element::u8);
        if (FLAGS_auto_resize) {
            inputTensorInfo.set_layout({ "NHWC" });
        }  else {
            inputTensorInfo.set_layout({ "NCHW" });
        }

        PreProcessSteps preProcessSteps;
        preProcessSteps.convert_element_type(ov::element::f32);
        if (FLAGS_auto_resize) {
            preProcessSteps.resize(ResizeAlgorithm::RESIZE_LINEAR);
        }

        InputNetworkInfo inputNetworkInfo;
        inputNetworkInfo.set_layout({ "NCHW" });

        InputInfo inputInfo;
        inputInfo.tensor(std::move(inputTensorInfo));
        inputInfo.preprocess(std::move(preProcessSteps));
        inputInfo.network(std::move(inputNetworkInfo));

        OutputTensorInfo outputTensorInfo;
        outputTensorInfo.set_element_type(ov::element::f32);

        OutputInfo outputInfo;
        outputInfo.tensor(std::move(outputTensorInfo));

        PrePostProcessor ppp;
        ppp.input(std::move(inputInfo));
        ppp.output(std::move(outputInfo));

        network = ppp.build(network);

//        net = ie_.LoadNetwork(network, deviceName, pluginConfig);
        m_net = m_core.compile_model(network, deviceName, pluginConfig);
        logExecNetworkInfo(m_net, xmlPath, deviceName, "Vehicle And License Plate Detection");

    }

//    InferenceEngine::InferRequest createInferRequest() {
//        return net.CreateInferRequest();
    ov::runtime::InferRequest createInferRequest() {
        return m_net.create_infer_request();
    }

//    void setImage(InferenceEngine::InferRequest& inferRequest, const cv::Mat& img) {
//        InferenceEngine::Blob::Ptr input = inferRequest.GetBlob(detectorInputBlobName);
//        if (InferenceEngine::Layout::NHWC == input->getTensorDesc().getLayout()) {
//            // autoResize is set
//            if (!img.isSubmatrix()) {
//                // just wrap Mat object with Blob::Ptr without additional memory allocation
//                InferenceEngine::Blob::Ptr frameBlob = wrapMat2Blob(img);
//                inferRequest.SetBlob(detectorInputBlobName, frameBlob);
//            } else {
//                throw std::logic_error("Sparse matrix are not supported");
//            }
//        } else {
//            matToBlob(img, input);
//        }
//    }
    void setImage(ov::runtime::InferRequest& inferRequest, const cv::Mat& img) {
        ov::runtime::Tensor input = inferRequest.get_tensor(detectorInputBlobName);
        if (3 == input.get_shape()[ov::layout::channels_idx(ov::Layout({ "NHWC" }))]) {
            // autoResize is set
            if (!img.isSubmatrix()) {
                // just wrap Mat object with Blob::Ptr without additional memory allocation
                ov::runtime::Tensor frameBlob = wrapMat2Tensor(img);
                inferRequest.set_tensor(detectorInputBlobName, frameBlob);
            } else {
                throw std::logic_error("Sparse matrix are not supported");
            }
        } else {
            matToTensor(img, input);
        }
    }

//    std::list<Result> getResults(InferenceEngine::InferRequest& inferRequest, cv::Size upscale, std::vector<std::string>& rawResults) {
    std::list<Result> getResults(ov::runtime::InferRequest& inferRequest, cv::Size upscale, std::vector<std::string>& rawResults) {
        // there is no big difference if InferReq of detector from another device is passed because the processing is the same for the same topology
        std::list<Result> results;
//        InferenceEngine::LockedMemory<const void> detectorOutputBlobMapped = InferenceEngine::as<
//            InferenceEngine::MemoryBlob>(inferRequest.GetBlob(detectorOutputBlobName))->rmap();
//        const float * const detections = detectorOutputBlobMapped.as<float *>();
        ov::runtime::Tensor tensor = inferRequest.get_tensor(detectorOutputBlobName);
        const float* const detections = tensor.data<float>();
        // pretty much regular SSD post-processing
        for (int i = 0; i < maxProposalCount; i++) {
            float image_id = detections[i * objectSize + 0];  // in case of batch
            if (image_id < 0) {  // indicates end of detections
                break;
            }
            auto label = static_cast<decltype(detectionTresholds.size())>(detections[i * objectSize + 1]);
            float confidence = detections[i * objectSize + 2];
            if (label - 1 < detectionTresholds.size() && confidence < detectionTresholds[label - 1]) {
                continue;
            }

            cv::Rect rect;
            rect.x = static_cast<int>(detections[i * objectSize + 3] * upscale.width);
            rect.y = static_cast<int>(detections[i * objectSize + 4] * upscale.height);
            rect.width = static_cast<int>(detections[i * objectSize + 5] * upscale.width) - rect.x;
            rect.height = static_cast<int>(detections[i * objectSize + 6] * upscale.height) - rect.y;
            results.push_back(Result{label, confidence, rect});
            std::ostringstream rawResultsStream;
            rawResultsStream << "[" << i << "," << label << "] element, prob = " << confidence
                        << "    (" << rect.x << "," << rect.y << ")-(" << rect.width << "," << rect.height << ")";
            rawResults.push_back(rawResultsStream.str());
        }
        return results;
    }

private:
    std::vector<float> detectionTresholds;
    std::string detectorInputBlobName;
    std::string detectorOutputBlobName;
//    InferenceEngine::Core ie_;  // The only reason to store a plugin as to assure that it lives at least as long as ExecutableNetwork
//    InferenceEngine::ExecutableNetwork net;
    ov::runtime::Core m_core;  // The only reason to store a plugin as to assure that it lives at least as long as ExecutableNetwork
    ov::runtime::ExecutableNetwork m_net;
};

class VehicleAttributesClassifier {
public:
    VehicleAttributesClassifier() = default;
//    VehicleAttributesClassifier(InferenceEngine::Core& ie, const std::string& deviceName,
//        const std::string& xmlPath, const bool autoResize, const std::map<std::string, std::string>& pluginConfig) : ie_(ie) {
    VehicleAttributesClassifier(ov::runtime::Core& core, const std::string& deviceName,
        const std::string& xmlPath, const bool autoResize, const std::map<std::string, std::string>& pluginConfig) : m_core(core) {
//        auto network = ie.ReadNetwork(FLAGS_m_va);
        auto network = m_core.read_model(FLAGS_m_va);

//        InferenceEngine::InputsDataMap attributesInputInfo(network.getInputsInfo());
//        if (attributesInputInfo.size() != 1) {
//            throw std::logic_error("Vehicle Attribs topology should have only one input");
//        }
//        InferenceEngine::InputInfo::Ptr& attributesInputInfoFirst = attributesInputInfo.begin()->second;
//        attributesInputInfoFirst->setPrecision(InferenceEngine::Precision::U8);
//        if (FLAGS_auto_resize) {
//            attributesInputInfoFirst->getPreProcess().setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
//            attributesInputInfoFirst->setLayout(InferenceEngine::Layout::NHWC);
//        } else {
//            attributesInputInfoFirst->setLayout(InferenceEngine::Layout::NCHW);
//        }

//        attributesInputName = attributesInputInfo.begin()->first;

        ov::OutputVector inputs = network->inputs();
        if (inputs.size() != 1) {
            throw std::logic_error("Vehicle Attribs topology should have only one input");
        }

        attributesInputName = network->input().get_any_name();

//        InferenceEngine::OutputsDataMap attributesOutputInfo(network.getOutputsInfo());
//        if (attributesOutputInfo.size() != 2) {
//            throw std::logic_error("Vehicle Attribs Network expects networks having two outputs");
//        }
//        auto it = attributesOutputInfo.begin();
//        it->second->setPrecision(InferenceEngine::Precision::FP32);
//        outputNameForColor = (it++)->second->getName();  // color is the first output
//        it->second->setPrecision(InferenceEngine::Precision::FP32);
//        outputNameForType = (it)->second->getName();  // type is the second output.

        ov::OutputVector outputs = network->outputs();
        if (outputs.size() != 2) {
            throw std::logic_error("Vehicle Attribs Network expects networks having two outputs");
        }

        // color is the first output
        outputNameForColor = outputs[0].get_any_name();
        // type is the second output.
        outputNameForType = outputs[1].get_any_name();

        InputTensorInfo inputTensorInfo;
        inputTensorInfo.set_element_type(ov::element::u8);
        if (FLAGS_auto_resize) {
            inputTensorInfo.set_layout({ "NHWC" });
        } else {
            inputTensorInfo.set_layout({ "NCHW" });
        }

        InputNetworkInfo inputNetworkInfo;
        inputNetworkInfo.set_layout({ "NCHW" });

        PreProcessSteps preProcessSteps;
        preProcessSteps.convert_element_type(ov::element::f32);
        if (FLAGS_auto_resize) {
            preProcessSteps.resize(ResizeAlgorithm::RESIZE_LINEAR);
        }

        InputInfo inputInfo;
        inputInfo.tensor(std::move(inputTensorInfo));
        inputInfo.preprocess(std::move(preProcessSteps));
        inputInfo.network(std::move(inputNetworkInfo));

        OutputTensorInfo outputTensorInfo;
        outputTensorInfo.set_element_type(ov::element::f32);

        OutputInfo outputColorInfo(outputNameForColor);
        outputColorInfo.tensor(std::move(outputTensorInfo));

        OutputInfo outputTypeInfo(outputNameForType);
        outputTypeInfo.tensor(std::move(outputTensorInfo));

        PrePostProcessor ppp;
        ppp.input(std::move(inputInfo));
        ppp.output(std::move(outputColorInfo));
        ppp.output(std::move(outputTypeInfo));

        network = ppp.build(network);

//        net = ie_.LoadNetwork(network, deviceName, pluginConfig);
//        logExecNetworkInfo(net, FLAGS_m_va, deviceName, "Vehicle Attributes Recognition");
        m_net = m_core.compile_model(network, deviceName, pluginConfig);
        logExecNetworkInfo(m_net, FLAGS_m_va, deviceName, "Vehicle Attributes Recognition");
    }

//    InferenceEngine::InferRequest createInferRequest() {
    ov::runtime::InferRequest createInferRequest() {
//        return net.CreateInferRequest();
        return m_net.create_infer_request();
    }

//    void setImage(InferenceEngine::InferRequest& inferRequest, const cv::Mat& img, const cv::Rect vehicleRect) {
//        InferenceEngine::Blob::Ptr roiBlob = inferRequest.GetBlob(attributesInputName);
//        if (InferenceEngine::Layout::NHWC == roiBlob->getTensorDesc().getLayout()) {  // autoResize is set
//            InferenceEngine::ROI cropRoi{0, static_cast<size_t>(vehicleRect.x), static_cast<size_t>(vehicleRect.y), static_cast<size_t>(vehicleRect.width),
//                static_cast<size_t>(vehicleRect.height)};
//            InferenceEngine::Blob::Ptr frameBlob = wrapMat2Blob(img);
//            InferenceEngine::Blob::Ptr roiBlob = make_shared_blob(frameBlob, cropRoi);
//            inferRequest.SetBlob(attributesInputName, roiBlob);
//        } else {
//            const cv::Mat& vehicleImage = img(vehicleRect);
//            matToBlob(vehicleImage, roiBlob);
//        }
//    }
    void setImage(ov::runtime::InferRequest& inferRequest, const cv::Mat& img, const cv::Rect vehicleRect) {
        ov::runtime::Tensor roiBlob = inferRequest.get_tensor(attributesInputName);
        if (3 == roiBlob.get_shape()[ov::layout::channels_idx(ov::Layout({ "NHWC" }))]) {
            // autoResize is set
            InferenceEngine::ROI cropRoi{
                0,
                static_cast<size_t>(vehicleRect.x), static_cast<size_t>(vehicleRect.y),
                static_cast<size_t>(vehicleRect.width), static_cast<size_t>(vehicleRect.height) };
            ov::runtime::Tensor frameBlob = wrapMat2Tensor(img);
            ov::Coordinate p00({ 0, cropRoi.posY, cropRoi.posX, 0 });
            ov::Coordinate p01({ 1, cropRoi.posY + cropRoi.sizeY, cropRoi.posX + cropRoi.sizeX, 3 });
            ov::runtime::Tensor roiBlob(frameBlob, p00, p01);

            inferRequest.set_tensor(attributesInputName, roiBlob);
        }
        else {
            const cv::Mat& vehicleImage = img(vehicleRect);
            matToTensor(vehicleImage, roiBlob);
        }
    }

//    std::pair<std::string, std::string> getResults(InferenceEngine::InferRequest& inferRequest) {
    std::pair<std::string, std::string> getResults(ov::runtime::InferRequest& inferRequest) {
        static const std::string colors[] = {
            "white", "gray", "yellow", "red", "green", "blue", "black"
        };
        static const std::string types[] = {
            "car", "van", "truck", "bus"
        };

//        // 7 possible colors for each vehicle and we should select the one with the maximum probability
//        InferenceEngine::LockedMemory<const void> colorsMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(
//            inferRequest.GetBlob(outputNameForColor))->rmap();
//        auto colorsValues = colorsMapped.as<float*>();
//        // 4 possible types for each vehicle and we should select the one with the maximum probability
//        InferenceEngine::LockedMemory<const void> typesMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(
//            inferRequest.GetBlob(outputNameForType))->rmap();
//        auto typesValues = typesMapped.as<float*>();

        // 7 possible colors for each vehicle and we should select the one with the maximum probability
        ov::runtime::Tensor colorsMapped = inferRequest.get_tensor(outputNameForColor);
        const float* colorsValues = colorsMapped.data<float>();

        // 4 possible types for each vehicle and we should select the one with the maximum probability
        ov::runtime::Tensor typesMapped = inferRequest.get_tensor(outputNameForType);
        const float* typesValues = typesMapped.data<float>();

        const auto color_id = std::max_element(colorsValues, colorsValues + 7) - colorsValues;
        const auto  type_id = std::max_element(typesValues,  typesValues  + 4) - typesValues;

        return std::pair<std::string, std::string>(colors[color_id], types[type_id]);
    }

private:
    std::string attributesInputName;
    std::string outputNameForColor;
    std::string outputNameForType;
//    InferenceEngine::Core ie_;  // The only reason to store a device is to assure that it lives at least as long as ExecutableNetwork
//    InferenceEngine::ExecutableNetwork net;
    ov::runtime::Core m_core;  // The only reason to store a device is to assure that it lives at least as long as ExecutableNetwork
    ov::runtime::ExecutableNetwork m_net;
};

class Lpr {
public:
    Lpr() = default;
//    Lpr(InferenceEngine::Core& ie, const std::string& deviceName, const std::string& xmlPath, const bool autoResize,
    Lpr(ov::runtime::Core& core, const std::string& deviceName, const std::string& xmlPath, const bool autoResize,
        const std::map<std::string, std::string>& pluginConfig) :
        m_core{core} {
//        auto network = ie.ReadNetwork(FLAGS_m_lpr);
        auto network = m_core.read_model(FLAGS_m_lpr);

        // LPR network should have 2 inputs (and second is just a stub) and one output
        // ---------------------------Check inputs ------------------------------------------------------
//        InferenceEngine::InputsDataMap LprInputInfo(network.getInputsInfo());
//        if (LprInputInfo.size() != 1 && LprInputInfo.size() != 2) {
//            throw std::logic_error("LPR should have 1 or 2 inputs");
//        }
//        InferenceEngine::InputInfo::Ptr& LprInputInfoFirst = LprInputInfo.begin()->second;
//        LprInputInfoFirst->setPrecision(InferenceEngine::Precision::U8);
//        if (FLAGS_auto_resize) {
//            LprInputInfoFirst->getPreProcess().setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
//            LprInputInfoFirst->setLayout(InferenceEngine::Layout::NHWC);
//        } else {
//            LprInputInfoFirst->setLayout(InferenceEngine::Layout::NCHW);
//        }
//        LprInputName = LprInputInfo.begin()->first;
//        if (LprInputInfo.size() == 2){
//            //LPR model that converted from Caffe have second a stub input
//            auto sequenceInput = (++LprInputInfo.begin());
//            LprInputSeqName = sequenceInput->first;
//        } else {
//            LprInputSeqName = "";
//        }

        ov::OutputVector inputs = network->inputs();
        if (inputs.size() != 1 && inputs.size() != 2) {
            throw std::logic_error("LPR should have 1 or 2 inputs");
        }

        for (auto input : inputs) {
            if (input.get_shape().size() == 4)
                LprInputName = input.get_any_name();
            if (input.get_shape().size() == 2)
                LprInputSeqName = input.get_any_name();
        }

//        LprInputName = inputs.size() == 4 ? inputs[0].get_any_name();
        // LPR model that converted from Caffe have second a stub input
//        LprInputSeqName = inputs.size() == 2 ? inputs[1].get_any_name() : "";

        // -----------------------------------------------------------------------------------------------------

        // ---------------------------Check outputs ------------------------------------------------------
//        InferenceEngine::OutputsDataMap LprOutputInfo(network.getOutputsInfo());
//        if (LprOutputInfo.size() != 1) {
//            throw std::logic_error("LPR should have 1 output");
//        }
//        LprOutputName = LprOutputInfo.begin()->first;
//        auto lprOutputInfo = (LprOutputInfo.begin());

        maxSequenceSizePerPlate = 1;
//        for (size_t dim : lprOutputInfo->second->getTensorDesc().getDims()) {
//            if (dim == 1) {
//                continue;
//            }
//            if (maxSequenceSizePerPlate == 1) {
//                maxSequenceSizePerPlate = dim;
//            } else {
//                throw std::logic_error("Every dimension of LPR output except for one must be of size 1");
//            }
//        }

        ov::OutputVector outputs = network->outputs();
        if (outputs.size() != 1) {
            throw std::logic_error("LPR should have 1 output");
        }

        LprOutputName = outputs[0].get_any_name();

        for (size_t dim : outputs[0].get_shape()) {
            if (dim == 1) {
                continue;
            }
            if (maxSequenceSizePerPlate == 1) {
                maxSequenceSizePerPlate = dim;
            } else {
                throw std::logic_error("Every dimension of LPR output except for one must be of size 1");
            }
        }

        InputTensorInfo inputTensorInfo;
        inputTensorInfo.set_element_type(ov::element::u8);
        if (FLAGS_auto_resize) {
            inputTensorInfo.set_layout({ "NHWC" });
        }
        else {
            inputTensorInfo.set_layout({ "NCHW" });
        }

        InputNetworkInfo inputNetworkInfo;
        inputNetworkInfo.set_layout({ "NCHW" });

        PreProcessSteps preProcessSteps;
        preProcessSteps.convert_element_type(ov::element::f32);
        if (FLAGS_auto_resize) {
            preProcessSteps.resize(ResizeAlgorithm::RESIZE_LINEAR);
        }

        InputInfo inputInfo(LprInputName);
        inputInfo.tensor(std::move(inputTensorInfo));
        inputInfo.preprocess(std::move(preProcessSteps));
        inputInfo.network(std::move(inputNetworkInfo));

        PrePostProcessor ppp;
        ppp.input(std::move(inputInfo));

        network = ppp.build(network);

//        net = ie_.LoadNetwork(network, deviceName, pluginConfig);
//        logExecNetworkInfo(net, FLAGS_m_lpr, deviceName, "License Plate Recognition");
        m_net = m_core.compile_model(network, deviceName, pluginConfig);
        logExecNetworkInfo(m_net, FLAGS_m_lpr, deviceName, "License Plate Recognition");
    }

//    InferenceEngine::InferRequest createInferRequest() {
//        return net.CreateInferRequest();
    ov::runtime::InferRequest createInferRequest() {
        return m_net.create_infer_request();
    }

//    void setImage(InferenceEngine::InferRequest& inferRequest, const cv::Mat& img, const cv::Rect plateRect) {
//        InferenceEngine::Blob::Ptr roiBlob = inferRequest.GetBlob(LprInputName);
//        if (InferenceEngine::Layout::NHWC == roiBlob->getTensorDesc().getLayout()) {  // autoResize is set
//            InferenceEngine::ROI cropRoi{0, static_cast<size_t>(plateRect.x), static_cast<size_t>(plateRect.y), static_cast<size_t>(plateRect.width),
//                static_cast<size_t>(plateRect.height)};
//            InferenceEngine::Blob::Ptr frameBlob = wrapMat2Blob(img);
//            InferenceEngine::Blob::Ptr roiBlob = make_shared_blob(frameBlob, cropRoi);
//            inferRequest.SetBlob(LprInputName, roiBlob);
//        } else {
//            const cv::Mat& vehicleImage = img(plateRect);
//            matToBlob(vehicleImage, roiBlob);
//        }
//
//        if (LprInputSeqName != "") {
//            InferenceEngine::Blob::Ptr seqBlob = inferRequest.GetBlob(LprInputSeqName);
//            InferenceEngine::LockedMemory<void> seqBlobMapped =
//                InferenceEngine::as<InferenceEngine::MemoryBlob>(seqBlob)->wmap();
//            float* blob_data = seqBlobMapped.as<float*>();
//            std::fill(blob_data, blob_data + seqBlob->getTensorDesc().getDims()[0], 1.0f);
//        }
//    }
    void setImage(ov::runtime::InferRequest& inferRequest, const cv::Mat& img, const cv::Rect plateRect) {
        ov::runtime::Tensor roiBlob = inferRequest.get_tensor(LprInputName);
        ov::Shape shape = roiBlob.get_shape();
        if ((shape.size() == 4) && (3 == shape[ov::layout::channels_idx(ov::Layout({ "NHWC" }))])) {
            // autoResize is set
            InferenceEngine::ROI cropRoi{
                0,
                static_cast<size_t>(plateRect.x), static_cast<size_t>(plateRect.y),
                static_cast<size_t>(plateRect.width), static_cast<size_t>(plateRect.height) };
            ov::runtime::Tensor frameBlob = wrapMat2Tensor(img);
//            InferenceEngine::Blob::Ptr roiBlob = make_shared_blob(frameBlob, cropRoi);
            ov::Coordinate p00({ 0, cropRoi.posY, cropRoi.posX, 0 });
            ov::Coordinate p01({ 1, cropRoi.posY + cropRoi.sizeY, cropRoi.posX + cropRoi.sizeX, 3 });
            ov::runtime::Tensor roiBlob(frameBlob, p00, p01);
            inferRequest.set_tensor(LprInputName, roiBlob);
        } else {
            const cv::Mat& vehicleImage = img(plateRect);
            matToTensor(vehicleImage, roiBlob);
        }

        if (LprInputSeqName != "") {
            ov::runtime::Tensor seqBlob = inferRequest.get_tensor(LprInputSeqName);
            float* blob_data = seqBlob.data<float>();
            std::fill(blob_data, blob_data + seqBlob.get_shape()[0], 1.0f);
        }
    }

//    std::string getResults(InferenceEngine::InferRequest& inferRequest) {
    std::string getResults(ov::runtime::InferRequest& inferRequest) {
        static const char* const items[] = {
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
        std::string result;
        result.reserve(14u + 6u);  // the longest province name + 6 plate signs
        // up to 88 items per license plate, ended with "-1"
//        InferenceEngine::LockedMemory<const void> lprOutputMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(
//            inferRequest.GetBlob(LprOutputName))->rmap();

//        InferenceEngine::Precision precision = inferRequest.GetBlob(LprOutputName)->getTensorDesc().getPrecision();

        ov::runtime::Tensor lprOutputMapped = inferRequest.get_tensor(LprOutputName);
        ov::element::Type precision = lprOutputMapped.get_element_type();

        switch (precision) {
//            case InferenceEngine::Precision::I32:
            case ov::element::i32:
            {
//                const auto data = lprOutputMapped.as<int32_t*>();
                const auto data = lprOutputMapped.data<int32_t>();
                for (int i = 0; i < maxSequenceSizePerPlate; i++) {
                    int32_t val = data[i];
                    if (val == -1) {
                        break;
                    }
                    result += items[val];
                }
            }
            break;

//            case InferenceEngine::Precision::FP32:
            case ov::element::f32:
            {
//                const auto data = lprOutputMapped.as<float*>();
                const auto data = lprOutputMapped.data<float>();
                for (int i = 0; i < maxSequenceSizePerPlate; i++) {
                    int32_t val = int32_t(data[i]);
                    if (val == -1) {
                        break;
                    }
                    result += items[val];
                }
            }
            break;

            default:
                throw std::logic_error("Not expected output blob precision");
                break;
        }
        return result;
    }

private:
    int maxSequenceSizePerPlate = 0;
    std::string LprInputName;
    std::string LprInputSeqName;
    std::string LprOutputName;
//    InferenceEngine::Core ie_;  // The only reason to store a device as to assure that it lives at least as long as ExecutableNetwork
//    InferenceEngine::ExecutableNetwork net;
    ov::runtime::Core m_core;  // The only reason to store a device as to assure that it lives at least as long as ExecutableNetwork
    ov::runtime::ExecutableNetwork m_net;
};
