// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>
#include <string>
#include <utility>
#include <vector>
#include <map>

#include <inference_engine.hpp>
#include <samples/common.hpp>
#include <samples/ocv_common.hpp>

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
    Detector(InferenceEngine::Core& ie, const std::string deviceName, const std::string& xmlPath, const std::vector<float>& detectionTresholds,
            const bool autoResize, const std::map<std::string, std::string> & pluginConfig) :
        detectionTresholds{detectionTresholds}, ie_{ie} {
        InferenceEngine::CNNNetReader netReader;
        netReader.ReadNetwork(xmlPath);
        std::string detectorBinFileName = fileNameNoExt(xmlPath) + ".bin";
        netReader.ReadWeights(detectorBinFileName);
        InferenceEngine::InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
        if (inputInfo.size() != 1) {
            throw std::logic_error("Detector should have only one input");
        }
        InferenceEngine::InputInfo::Ptr& inputInfoFirst = inputInfo.begin()->second;
        inputInfoFirst->setPrecision(InferenceEngine::Precision::U8);
        if (autoResize) {
            inputInfoFirst->getPreProcess().setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
            inputInfoFirst->setLayout(InferenceEngine::Layout::NHWC);
        } else {
            inputInfoFirst->setLayout(InferenceEngine::Layout::NCHW);
        }

        detectorInputBlobName = inputInfo.begin()->first;

        // ---------------------------Check outputs ------------------------------------------------------
        InferenceEngine::OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
        if (outputInfo.size() != 1) {
            throw std::logic_error("Vehicle Detection network should have only one output");
        }
        InferenceEngine::DataPtr& _output = outputInfo.begin()->second;
        const InferenceEngine::SizeVector outputDims = _output->getTensorDesc().getDims();
        detectorOutputBlobName = outputInfo.begin()->first;
        if (maxProposalCount != outputDims[2]) {
            throw std::logic_error("unexpected ProposalCount");
        }
        if (objectSize != outputDims[3]) {
            throw std::logic_error("Output should have 7 as a last dimension");
        }
        if (outputDims.size() != 4) {
            throw std::logic_error("Incorrect output dimensions for SSD");
        }
        _output->setPrecision(InferenceEngine::Precision::FP32);

        net = ie_.LoadNetwork(netReader.getNetwork(), deviceName, pluginConfig);
    }

    InferenceEngine::InferRequest createInferRequest() {
        return net.CreateInferRequest();
    }

    void setImage(InferenceEngine::InferRequest& inferRequest, const cv::Mat& img) {
        InferenceEngine::Blob::Ptr input = inferRequest.GetBlob(detectorInputBlobName);
        if (InferenceEngine::Layout::NHWC == input->getTensorDesc().getLayout()) {  // autoResize is set
            if (!img.isSubmatrix()) {
                // just wrap Mat object with Blob::Ptr without additional memory allocation
                InferenceEngine::Blob::Ptr frameBlob = wrapMat2Blob(img);
                inferRequest.SetBlob(detectorInputBlobName, frameBlob);
            } else {
                throw std::logic_error("Sparse matrix are not supported");
            }
        } else {
            matU8ToBlob<uint8_t>(img, input);
        }
    }

    std::list<Result> getResults(InferenceEngine::InferRequest& inferRequest, cv::Size upscale, std::ostream* rawResults = nullptr) {
        // there is no big difference if InferReq of detector from another device is passed because the processing is the same for the same topology
        std::list<Result> results;
        const float * const detections = inferRequest.GetBlob(detectorOutputBlobName)->buffer().as<float *>();
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

            if (rawResults) {
                *rawResults << "[" << i << "," << label << "] element, prob = " << confidence
                            << "    (" << rect.x << "," << rect.y << ")-(" << rect.width << "," << rect.height << ")" << std::endl;
            }
        }
        return results;
    }

private:
    std::vector<float> detectionTresholds;
    std::string detectorInputBlobName;
    std::string detectorOutputBlobName;
    InferenceEngine::Core ie_;  // The only reason to store a plugin as to assure that it lives at least as long as ExecutableNetwork
    InferenceEngine::ExecutableNetwork net;
};

class VehicleAttributesClassifier {
public:
    VehicleAttributesClassifier() = default;
    VehicleAttributesClassifier(InferenceEngine::Core& ie, const std::string & deviceName,
        const std::string& xmlPath, const bool autoResize, const std::map<std::string, std::string> & pluginConfig) : ie_(ie) {
        InferenceEngine::CNNNetReader attributesNetReader;
        attributesNetReader.ReadNetwork(FLAGS_m_va);
        std::string attributesBinFileName = fileNameNoExt(FLAGS_m_va) + ".bin";
        attributesNetReader.ReadWeights(attributesBinFileName);
        InferenceEngine::InputsDataMap attributesInputInfo(attributesNetReader.getNetwork().getInputsInfo());
        if (attributesInputInfo.size() != 1) {
            throw std::logic_error("Vehicle Attribs topology should have only one input");
        }
        InferenceEngine::InputInfo::Ptr& attributesInputInfoFirst = attributesInputInfo.begin()->second;
        attributesInputInfoFirst->setPrecision(InferenceEngine::Precision::U8);
        if (FLAGS_auto_resize) {
            attributesInputInfoFirst->getPreProcess().setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
            attributesInputInfoFirst->setLayout(InferenceEngine::Layout::NHWC);
        } else {
            attributesInputInfoFirst->setLayout(InferenceEngine::Layout::NCHW);
        }

        attributesInputName = attributesInputInfo.begin()->first;

        InferenceEngine::OutputsDataMap attributesOutputInfo(attributesNetReader.getNetwork().getOutputsInfo());
        if (attributesOutputInfo.size() != 2) {
            throw std::logic_error("Vehicle Attribs Network expects networks having two outputs");
        }
        auto it = attributesOutputInfo.begin();
        it->second->setPrecision(InferenceEngine::Precision::FP32);
        outputNameForColor = (it++)->second->getName();  // color is the first output
        it->second->setPrecision(InferenceEngine::Precision::FP32);
        outputNameForType = (it)->second->getName();  // type is the second output.

        net = ie_.LoadNetwork(attributesNetReader.getNetwork(), deviceName, pluginConfig);
    }

    InferenceEngine::InferRequest createInferRequest() {
        return net.CreateInferRequest();
    }

    void setImage(InferenceEngine::InferRequest& inferRequest, const cv::Mat& img, const cv::Rect vehicleRect) {
        InferenceEngine::Blob::Ptr roiBlob = inferRequest.GetBlob(attributesInputName);
        if (InferenceEngine::Layout::NHWC == roiBlob->getTensorDesc().getLayout()) {  // autoResize is set
            InferenceEngine::ROI cropRoi{0, static_cast<size_t>(vehicleRect.x), static_cast<size_t>(vehicleRect.y), static_cast<size_t>(vehicleRect.width),
                static_cast<size_t>(vehicleRect.height)};
            InferenceEngine::Blob::Ptr frameBlob = wrapMat2Blob(img);
            InferenceEngine::Blob::Ptr roiBlob = make_shared_blob(frameBlob, cropRoi);
            inferRequest.SetBlob(attributesInputName, roiBlob);
        } else {
            const cv::Mat& vehicleImage = img(vehicleRect);
            matU8ToBlob<uint8_t>(vehicleImage, roiBlob);
        }
    }
    std::pair<std::string, std::string> getResults(InferenceEngine::InferRequest& inferRequest) {
        static const std::string colors[] = {
            "white", "gray", "yellow", "red", "green", "blue", "black"
        };
        static const std::string types[] = {
            "car", "van", "truck", "bus"
        };

        // 7 possible colors for each vehicle and we should select the one with the maximum probability
        auto colorsValues = inferRequest.GetBlob(outputNameForColor)->buffer().as<float*>();
        // 4 possible types for each vehicle and we should select the one with the maximum probability
        auto typesValues  = inferRequest.GetBlob(outputNameForType)->buffer().as<float*>();

        const auto color_id = std::max_element(colorsValues, colorsValues + 7) - colorsValues;
        const auto  type_id = std::max_element(typesValues,  typesValues  + 4) - typesValues;
        return std::pair<std::string, std::string>(colors[color_id], types[type_id]);
    }

private:
    std::string attributesInputName;
    std::string outputNameForColor;
    std::string outputNameForType;
    InferenceEngine::Core ie_;  // The only reason to store a device is to assure that it lives at least as long as ExecutableNetwork
    InferenceEngine::ExecutableNetwork net;
};

class Lpr {
public:
    Lpr() = default;
    Lpr(InferenceEngine::Core& ie, const std::string & deviceName, const std::string& xmlPath, const bool autoResize,
        const std::map<std::string, std::string> &pluginConfig) :
        ie_{ie} {
        InferenceEngine::CNNNetReader LprNetReader;
        LprNetReader.ReadNetwork(FLAGS_m_lpr);
        std::string lprBinFileName = fileNameNoExt(FLAGS_m_lpr) + ".bin";
        LprNetReader.ReadWeights(lprBinFileName);

        /** LPR network should have 2 inputs (and second is just a stub) and one output **/
        // ---------------------------Check inputs ------------------------------------------------------
        InferenceEngine::InputsDataMap LprInputInfo(LprNetReader.getNetwork().getInputsInfo());
        if (LprInputInfo.size() != 2) {
            throw std::logic_error("LPR should have 2 inputs");
        }
        InferenceEngine::InputInfo::Ptr& LprInputInfoFirst = LprInputInfo.begin()->second;
        LprInputInfoFirst->setPrecision(InferenceEngine::Precision::U8);
        if (FLAGS_auto_resize) {
            LprInputInfoFirst->getPreProcess().setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
            LprInputInfoFirst->setLayout(InferenceEngine::Layout::NHWC);
        } else {
            LprInputInfoFirst->setLayout(InferenceEngine::Layout::NCHW);
        }
        LprInputName = LprInputInfo.begin()->first;
        auto sequenceInput = (++LprInputInfo.begin());
        LprInputSeqName = sequenceInput->first;
        maxSequenceSizePerPlate = sequenceInput->second->getTensorDesc().getDims()[0];
        // -----------------------------------------------------------------------------------------------------

        // ---------------------------Check outputs ------------------------------------------------------
        InferenceEngine::OutputsDataMap LprOutputInfo(LprNetReader.getNetwork().getOutputsInfo());
        if (LprOutputInfo.size() != 1) {
            throw std::logic_error("LPR should have 1 output");
        }
        LprOutputName = LprOutputInfo.begin()->first;

        net = ie_.LoadNetwork(LprNetReader.getNetwork(), deviceName, pluginConfig);
    }

    InferenceEngine::InferRequest createInferRequest() {
        return net.CreateInferRequest();
    }

    void setImage(InferenceEngine::InferRequest& inferRequest, const cv::Mat& img, const cv::Rect plateRect) {
        InferenceEngine::Blob::Ptr roiBlob = inferRequest.GetBlob(LprInputName);
        if (InferenceEngine::Layout::NHWC == roiBlob->getTensorDesc().getLayout()) {  // autoResize is set
            InferenceEngine::ROI cropRoi{0, static_cast<size_t>(plateRect.x), static_cast<size_t>(plateRect.y), static_cast<size_t>(plateRect.width),
                static_cast<size_t>(plateRect.height)};
            InferenceEngine::Blob::Ptr frameBlob = wrapMat2Blob(img);
            InferenceEngine::Blob::Ptr roiBlob = make_shared_blob(frameBlob, cropRoi);
            inferRequest.SetBlob(LprInputName, roiBlob);
        } else {
            const cv::Mat& vehicleImage = img(plateRect);
            matU8ToBlob<uint8_t>(vehicleImage, roiBlob);
        }
        InferenceEngine::Blob::Ptr seqBlob = inferRequest.GetBlob(LprInputSeqName);
        // second input is sequence, which is some relic from the training
        // it should have the leading 0.0f and rest 1.0f
        float* blob_data = seqBlob->buffer().as<float*>();
        blob_data[0] = 0.0f;
        std::fill(blob_data + 1, blob_data + maxSequenceSizePerPlate, 1.0f);
    }

    std::string getResults(InferenceEngine::InferRequest& inferRequest) {
        static const std::vector<std::string> items = {
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
        const auto data = inferRequest.GetBlob(LprOutputName)->buffer().as<float*>();
        for (int i = 0; i < maxSequenceSizePerPlate; i++) {
            if (data[i] == -1) {
                break;
            }
            result += items[static_cast<std::vector<std::string>::size_type>(data[i])];
        }
        return result;
    }

private:
    int maxSequenceSizePerPlate;
    std::string LprInputName;
    std::string LprInputSeqName;
    std::string LprOutputName;
    InferenceEngine::Core ie_;  // The only reason to store a device as to assure that it lives at least as long as ExecutableNetwork
    InferenceEngine::ExecutableNetwork net;
};
