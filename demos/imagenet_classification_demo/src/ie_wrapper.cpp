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
                     const std::string& deviceName, size_t batchSize):
           modelPath(modelPath), deviceName(deviceName), ie(ie) {
    netReader.ReadNetwork(modelPath);
    std::string binFileName = fileNameNoExt(modelPath) + ".bin";
    netReader.ReadWeights(binFileName);
    network = netReader.getNetwork();
    
    resizeNetwork(batchSize);

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

    if (deviceName.find("CPU") != std::string::npos) {
            /**
             * cpu_extensions library is compiled from "extension" folder containing
             * custom MKLDNNPlugin layer implementations. These layers are not supported
             * by mkldnn, but they can be useful for inferring custom topologies.
            **/
        ie.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>(), "CPU");
    }

    executableNetwork = ie.LoadNetwork(network, deviceName);
    request = executableNetwork.CreateInferRequest();
}



void IEWrapper::setInputBlob(const std::string& blobName,
                             const std::vector<cv::Mat>& images,
                             int firstIndex) {
    auto blobDims = inputBlobsDimsInfo[blobName];
    if (blobDims.size() != 4) {
        throw std::runtime_error("Input data does not match size of the blob");
    }
    
    auto inputBlob = request.GetBlob(blobName);
    size_t batchSize = network.getBatchSize();
    size_t imgDataSize = images.size();
    
    for(size_t i = 0; i < batchSize; i++) {        
        cv::Mat inputImg = images.at((firstIndex+i)%imgDataSize);
        matU8ToBlob<uint8_t>(inputImg, inputBlob, i);
        //<PrecisionTrait<Precision::U8>::value_type>
    }
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

void IEWrapper::resizeNetwork(size_t batchSize) { //OK ... ?
    auto input_shapes = network.getInputShapes();
    std::string input_name;
    SizeVector input_shape;
    std::tie(input_name, input_shape) = *input_shapes.begin();
    input_shape[0] = batchSize;
    input_shape[2] = input_shapes[input_name][2];
    input_shape[3] = input_shapes[input_name][3];
    input_shapes[input_name] = input_shape;
    std::cout << "Resizing network to the image size = [" << input_shapes[input_name][2] << "x" << input_shapes[input_name][3] << "] "
              << "with batch = " << batchSize << std::endl;
    network.reshape(input_shapes);
}

void IEWrapper::setBatchSize(size_t size) {
    network.setBatchSize(size);
}

size_t IEWrapper::getBatchSize() {
    return network.getBatchSize();
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

void IEWrapper::startAsync(){
    request.StartAsync();
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
