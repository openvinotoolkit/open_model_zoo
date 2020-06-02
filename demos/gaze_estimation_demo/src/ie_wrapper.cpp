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
    network = ie.ReadNetwork(modelPath);
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
    matU8ToBlob<uint8_t>(resizedImage, inputBlob);
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
    LockedMemory<void> blobMapped = as<MemoryBlob>(request.GetBlob(blobName))->wmap();
    auto buffer = blobMapped.as<float *>();
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
    
    LockedMemory<const void> blobMapped = as<MemoryBlob>(request.GetBlob(blobName))->rmap();
    auto buffer = blobMapped.as<float *>();

    for (int i = 0; i < dataSize; ++i) {
        output.push_back(buffer[i]);
    }
}

const std::map<std::string, std::vector<unsigned long>>& IEWrapper::getInputBlobDimsInfo() const {
    return inputBlobsDimsInfo;
}
const std::map<std::string, std::vector<unsigned long>>& IEWrapper::getOutputBlobDimsInfo() const {
    return outputBlobsDimsInfo;
}

std::string IEWrapper::expectSingleInput() const {
    if (inputBlobsDimsInfo.size() != 1) {
        throw std::runtime_error(modelPath + ": expected to have 1 input");
    }

    return inputBlobsDimsInfo.begin()->first;
}

std::string IEWrapper::expectSingleOutput() const {
    if (outputBlobsDimsInfo.size() != 1) {
        throw std::runtime_error(modelPath + ": expected to have 1 output");
    }

    return outputBlobsDimsInfo.begin()->first;
}

void IEWrapper::expectImageInput(const std::string& blobName) const {
    const auto& dims = inputBlobsDimsInfo.at(blobName);

    if (dims.size() != 4 || dims[0] != 1 || dims[1] != 3) {
        throw std::runtime_error(modelPath + ": expected \"" + blobName + "\" to have dimensions 1x3xHxW");
    }
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
