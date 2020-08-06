/*
// Copyright (C) 2018-2019 Intel Corporation
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
#include "detection_pipeline.h"
#include <samples/args_helper.hpp>

using namespace InferenceEngine;

DetectionPipeline::DetectionPipeline(){
}


DetectionPipeline::~DetectionPipeline(){
}

void DetectionPipeline::init(const std::string& model_name, const CnnConfig& cnnConfig, float confidenceThreshold, bool useAutoResize, InferenceEngine::Core* engine){
    PipelineBase::init(model_name, cnnConfig, engine);

    this->useAutoResize = useAutoResize;
    this->confidenceThreshold = confidenceThreshold;

    // --- Setting image info for every request in a pool. We can do it once and reuse this info at every submit -------
    if (!imageInfoInputName.empty()) {
        for (auto &pair : requestsPool) {
            auto blob = pair.first->GetBlob(imageInfoInputName);
            LockedMemory<void> blobMapped = as<MemoryBlob>(blob)->wmap();
            auto data = blobMapped.as<float *>();
            data[0] = static_cast<float>(netInputHeight);  // height
            data[1] = static_cast<float>(netInputWidth);  // width
            data[2] = 1;
        }
    }
}

void DetectionPipeline::PrepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork){
    // --------------------------- Configure input & output ---------------------------------------------
    // --------------------------- Prepare input blobs -----------------------------------------------------
    slog::info << "Checking that the inputs are as the demo expects" << slog::endl;
    InputsDataMap inputInfo(cnnNetwork.getInputsInfo());

    for (const auto & inputInfoItem : inputInfo) {
        if (inputInfoItem.second->getTensorDesc().getDims().size() == 4) {  // 1st input contains images
            imageInputName = inputInfoItem.first;
            inputInfoItem.second->setPrecision(Precision::U8);
            if (useAutoResize) {
                inputInfoItem.second->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
                inputInfoItem.second->getInputData()->setLayout(Layout::NHWC);
            }
            else {
                inputInfoItem.second->getInputData()->setLayout(Layout::NCHW);
            }
            const TensorDesc& inputDesc = inputInfoItem.second->getTensorDesc();
            netInputHeight = getTensorHeight(inputDesc);
            netInputWidth = getTensorWidth(inputDesc);
        }
        else if (inputInfoItem.second->getTensorDesc().getDims().size() == 2) {  // 2nd input contains image info
            imageInfoInputName = inputInfoItem.first;
            inputInfoItem.second->setPrecision(Precision::FP32);
        }
        else {
            throw std::logic_error("Unsupported " +
                std::to_string(inputInfoItem.second->getTensorDesc().getDims().size()) + "D "
                "input layer '" + inputInfoItem.first + "'. "
                "Only 2D and 4D input layers are supported");
        }
    }

    // --------------------------- Prepare output blobs -----------------------------------------------------
    slog::info << "Checking that the outputs are as the demo expects" << slog::endl;
    OutputsDataMap outputInfo(cnnNetwork.getOutputsInfo());
    if (outputInfo.size() != 1) {
        throw std::logic_error("This demo accepts networks having only one output");
    }
    DataPtr& output = outputInfo.begin()->second;
    outputName = outputInfo.begin()->first;

    int num_classes = 0;

    if (auto ngraphFunction = cnnNetwork.getFunction()) {
        for (const auto op : ngraphFunction->get_ops()) {
            if (op->get_friendly_name() == outputName) {
                auto detOutput = std::dynamic_pointer_cast<ngraph::op::DetectionOutput>(op);
                if (!detOutput) {
                    THROW_IE_EXCEPTION << "Object Detection network output layer(" + op->get_friendly_name() +
                        ") should be DetectionOutput, but was " + op->get_type_info().name;
                }

                num_classes = detOutput->get_attrs().num_classes;
                break;
            }
        }
    }
    else {
        throw std::logic_error("This demo requires IR version no older than 10");
    }
    if (labels.size()){
        if (static_cast<int>(labels.size()) == (num_classes - 1)) {  // if network assumes default "background" class, having no label
            labels.insert(labels.begin(), "fake");
        }
        else if (static_cast<int>(labels.size()) != num_classes) {
            throw std::logic_error("The number of labels is different from numbers of model classes");
        }
    }

    const SizeVector outputDims = output->getTensorDesc().getDims();
    maxProposalCount = outputDims[2];
    objectSize = outputDims[3];
    if (objectSize != 7) {
        throw std::logic_error("Output should have 7 as a last dimension");
    }
    if (outputDims.size() != 4) {
        throw std::logic_error("Incorrect output dimensions for SSD");
    }
    output->setPrecision(Precision::FP32);
    output->setLayout(Layout::NCHW);
}

int64_t DetectionPipeline::submitImage(cv::Mat img){
    auto request = getIdleRequest();
    if (!request)
        return -1;

    if (useAutoResize) {
        /* Just set input blob containing read image. Resize and layout conversion will be done automatically */
        request->SetBlob(imageInputName, wrapMat2Blob(img));
    }
    else {
        /* Resize and copy data from the image to the input blob */
        Blob::Ptr frameBlob = request->GetBlob(imageInputName);
        matU8ToBlob<uint8_t>(img, frameBlob);
    }

    return submitRequest(request);
}

DetectionPipeline::DetectionResults DetectionPipeline::getDetectionResults(){
    auto reqResult = PipelineBase::getResult();
    if (reqResult.IsEmpty()){
        return DetectionResults();
    }

    LockedMemory<const void> outputMapped = reqResult.output->rmap();
    const float *detections = outputMapped.as<float*>();

    DetectionResults results;
    results.frameId = reqResult.frameId;

    for (size_t i = 0; i < maxProposalCount; i++) {
        ObjectDesc desc;

        float image_id = detections[i * objectSize + 0];
        if (image_id < 0) {
            break;
        }

        desc.confidence = detections[i * objectSize + 2];
        desc.labelID = static_cast<int>(detections[i * objectSize + 1]);
        desc.label = desc.labelID < labels.size() ? labels[desc.labelID] : std::string("Label #") + std::to_string(desc.labelID);
        desc.x = detections[i * objectSize + 3];
        desc.y = detections[i * objectSize + 4];
        desc.width = detections[i * objectSize + 5] - desc.x;
        desc.height = detections[i * objectSize + 6] - desc.y;

        if (false) {
            std::cout << "[" << i << "," << desc.labelID << "] element, prob = " << desc.confidence <<
                "    (" << desc.x << "," << desc.y << ")-(" << desc.br().x << "," << desc.br().y << ")"
                << ((desc.confidence > confidenceThreshold) ? " WILL BE RENDERED!" : "") << std::endl;
        }

        if (desc.confidence > confidenceThreshold) {
            /** Filtering out objects with confidence < confidence_threshold probability **/
            results.objects.push_back(desc);
        }
    }

    return results;
}

void DetectionPipeline::loadLabels(const std::string & labelFilename){
    /** Read labels (if any)**/
    if (!labelFilename.empty()) {
        std::ifstream inputFile(labelFilename);
        std::string label;
        while (std::getline(inputFile, label)) {
            labels.push_back(label);
        }
        if (labels.empty())
            throw std::logic_error("File empty or not found: " + labelFilename);
    }
}
