// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <map>
#include <string>
#include <vector>

#include "ie_wrapper.hpp"

using namespace InferenceEngine;

namespace gaze_estimation {

IEWrapper::IEWrapper(InferenceEngine::Core& ie,
                     const std::string& modelPath,
                     const std::string& deviceName):
           modelPath(modelPath), deviceName(deviceName), ie(ie) {
    netReader.ReadNetwork(modelPath);
    std::string binFileName = fileNameNoExt(modelPath) + ".bin";
    netReader.ReadWeights(binFileName);
    network = netReader.getNetwork();
    setExecPart();
}

void IEWrapper::setExecPart() {
    // set map of input blob name -- blob dimension pairs
    auto inputInfo = network.getInputsInfo();
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

    executableNetwork = ie.LoadNetwork(network, deviceName);
    request = executableNetwork.CreateInferRequest();
}

void IEWrapper::setInputBlob(const std::string& blobName,
                             const cv::Mat& image) {
    auto blobDims = inputBlobsDimsInfo[blobName];

    if (blobDims.size() != 4) {
        throw std::runtime_error("Input data does not match size of the blob");
    }

    auto scaledSize = cv::Size(static_cast<int>(blobDims[3]), static_cast<int>(blobDims[2]));
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, scaledSize, 0, 0, cv::INTER_CUBIC);

    auto inputBlob = request.GetBlob(blobName);
    matU8ToBlob<PrecisionTrait<Precision::U8>::value_type>(resizedImage, inputBlob);
}

void IEWrapper::setInputBlob(const std::string& blobName,
                             const std::vector<float>& data) {
    auto blobDims = inputBlobsDimsInfo[blobName];
    unsigned long dimsProduct = 1;
    for (auto const& dim : blobDims) {
        dimsProduct *= dim;
    }
    if (dimsProduct != data.size()) {
        throw std::runtime_error("Input data does not match size of the blob");
    }
    auto inputBlob = request.GetBlob(blobName);
    auto buffer = inputBlob->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type *>();
    for (unsigned long int i = 0; i < data.size(); ++i) {
        buffer[i] = data[i];
    }
}

void IEWrapper::getOutputBlob(const std::string& blobName,
                              std::vector<float> &output) {
    output.clear();
    auto blobDims = outputBlobsDimsInfo[blobName];
    auto dataSize = 1;
    for (auto dim : blobDims) {
        dataSize *= dim;
    }
    auto outputBlob = request.GetBlob(blobName);
    auto buffer = outputBlob->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type *>();

    for (int i = 0; i < dataSize; ++i) {
        output.push_back(buffer[i]);
    }
}

void IEWrapper::getOutputBlob(std::vector<float>& output) {
    output.clear();
    auto blobName = outputBlobsDimsInfo.begin()->first;
    auto blobDims = outputBlobsDimsInfo[blobName];
    auto dataSize = 1;
    for (auto const& dim : blobDims) {
        dataSize *= dim;
    }
    auto outputBlob = request.GetBlob(blobName);
    auto buffer = outputBlob->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type *>();

    for (int i = 0; i < dataSize; ++i) {
        output.push_back(buffer[i]);
    }
}

const std::map<std::string, std::vector<unsigned long>>& IEWrapper::getIputBlobDimsInfo() const {
    return inputBlobsDimsInfo;
}
const std::map<std::string, std::vector<unsigned long>>& IEWrapper::getOutputBlobDimsInfo() const {
    return outputBlobsDimsInfo;
}

void IEWrapper::infer() {
    request.Infer();
}

void IEWrapper::reshape(const std::map<std::string, std::vector<unsigned long> > &newBlobsDimsInfo) {
    if (inputBlobsDimsInfo.size() != newBlobsDimsInfo.size()) {
        throw std::runtime_error("Mismatch in the number of blobs being reshaped");
    }

    auto inputShapes = network.getInputShapes();
    for (auto it = newBlobsDimsInfo.begin(); it != newBlobsDimsInfo.end(); ++it) {
        auto blobName = it->first;
        auto blobDims = it->second;

        InferenceEngine::SizeVector blobDims_(blobDims.data(), blobDims.data() + blobDims.size());
        inputShapes[blobName] = blobDims_;
    }
    network.reshape(inputShapes);
    setExecPart();
}

void IEWrapper::printPerlayerPerformance() const {
    std::cout << "\n-----------------START-----------------" << std::endl;
    std::cout << "Performance for " << modelPath << " model\n" << std::endl;
    printPerformanceCounts(request, std::cout, getFullDeviceName(ie, deviceName), false);
    std::cout << "------------------END------------------\n" << std::endl;
}
}  // namespace gaze_estimation
