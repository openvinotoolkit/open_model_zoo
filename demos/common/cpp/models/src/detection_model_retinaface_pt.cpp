/*
// Copyright (C) 2020-2021 Intel Corporation
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

#include <ngraph/ngraph.hpp>
#include <utils/common.hpp>
#include <utils/slog.hpp>
#include "models/detection_model_retinaface_pt.h"

ModelRetinaFacePT::ModelRetinaFacePT(const std::string& modelFileName, float confidenceThreshold, bool useAutoResize, float boxIOUThreshold)
    : DetectionModel(modelFileName, confidenceThreshold, useAutoResize, {"Face"}),  // Default label is "Face"
    landmarksNum(0), boxIOUThreshold(boxIOUThreshold) {
}

void ModelRetinaFacePT::prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) {
    // --------------------------- Configure input & output -------------------------------------------------
    // --------------------------- Prepare input blobs ------------------------------------------------------
    slog::info << "Checking that the inputs are as the demo expects" << slog::endl;
    InferenceEngine::InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
    if (inputInfo.size() != 1) {
        throw std::logic_error("This demo accepts networks that have only one input");
    }
    InferenceEngine::InputInfo::Ptr& input = inputInfo.begin()->second;
    std::string imageInputName = inputInfo.begin()->first;
    inputsNames.push_back(imageInputName);
    input->setPrecision(InferenceEngine::Precision::U8);
    if (useAutoResize) {
        input->getPreProcess().setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
        input->getInputData()->setLayout(InferenceEngine::Layout::NHWC);
    }
    else {
        input->getInputData()->setLayout(InferenceEngine::Layout::NCHW);
    }

    //--- Reading image input parameters
    imageInputName = inputInfo.begin()->first;
    const InferenceEngine::TensorDesc& inputDesc = inputInfo.begin()->second->getTensorDesc();
    netInputHeight = getTensorHeight(inputDesc);
    netInputWidth = getTensorWidth(inputDesc);

    // --------------------------- Prepare output blobs -----------------------------------------------------
    slog::info << "Checking that the outputs are as the demo expects" << slog::endl;

    InferenceEngine::OutputsDataMap outputInfo(cnnNetwork.getOutputsInfo());
    landmarksNum = 0;

    std::vector<uint32_t> outputsSizes[OT_MAX];
    for (auto& output : outputInfo) {
        output.second->setPrecision(InferenceEngine::Precision::FP32);
        output.second->setLayout(output.second->getDims().size()==4 ? InferenceEngine::Layout::NCHW : InferenceEngine::Layout::CHW);
        outputsNames.push_back(output.first);

        outputsNames.resize(2);
        if (output.first.find("bbox") != std::string::npos) {
            outputsNames[OT_BBOX] = output.first;
        }
        else if (output.first.find("cls") != std::string::npos) {
            outputsNames[OT_SCORES] = output.first;
        }
        else if (output.first.find("landmark") != std::string::npos) {
            // Landmarks might be optional, if it is present, resize names array to fit landmarks output name to the last item of array
            // Considering that other outputs names are already filled in or will be filled later
            outputsNames.resize(std::max(outputsNames.size(), (size_t)OT_LANDMARK + 1));
            outputsNames[OT_LANDMARK] = output.first;
            landmarksNum = output.second->getDims()[2] / 2; // Each landmark consist of 2 variables (x and y)
        }
        else {
            continue;
        }
    }

    if (outputsNames[OT_BBOX] == "" || outputsNames[OT_SCORES] == "") {
        throw std::runtime_error("Bbox or cls layers are not found");
    }

    priors = generatePriorData();
}


std::vector<size_t> ModelRetinaFacePT::filterByScore(const InferenceEngine::MemoryBlob::Ptr& rawData, const float confidenceThreshold) {
    std::vector<size_t> indicies;
    auto desc = rawData->getTensorDesc();
    auto sz = desc.getDims();
    InferenceEngine::LockedMemory<const void> outputMapped = rawData->rmap();
    const float *memPtr = outputMapped.as<float*>();

    for (size_t x = 0; x < sz[1]; ++x) {
        auto idx = (x * sz[2] + 1);
        auto score = memPtr[idx];
        if (score >= confidenceThreshold) {
            indicies.push_back(x);
        }
    }

    return indicies;
}

std::vector<float> ModelRetinaFacePT::getFilteredScores(const InferenceEngine::MemoryBlob::Ptr& rawData, const std::vector<size_t>& indicies) {
    InferenceEngine::LockedMemory<const void> outputMapped = rawData->rmap();
    const float *memPtr = outputMapped.as<float*>();
    auto desc = rawData->getTensorDesc();
    auto sz = desc.getDims();

    std::vector<float> scores;
    scores.reserve(indicies.size());

    for (auto i : indicies) {
        scores.push_back(memPtr[i*sz[2] + 1]);
    }
    return scores;
}

std::vector<cv::Point2f> ModelRetinaFacePT::getFilteredLandmarks(const InferenceEngine::MemoryBlob::Ptr& rawData, const std::vector<size_t>& indicies, int imgWidth, int imgHeight) {
    InferenceEngine::LockedMemory<const void> outputMapped = rawData->rmap();
    const float *memPtr = outputMapped.as<float*>();
    auto desc = rawData->getTensorDesc();
    auto sz = desc.getDims();

    std::vector<cv::Point2f> landmarks(landmarksNum*indicies.size());

    for (size_t i = 0; i < indicies.size(); i++) {
        size_t idx = indicies[i];
        auto& prior = priors[idx];
        for (size_t j = 0; j < landmarksNum; j++) {
            landmarks[i*landmarksNum + j].x = (prior.cX + memPtr[idx*sz[2] + j*2] * variance[0] * prior.width) * imgWidth;
            landmarks[i*landmarksNum + j].y = (prior.cY + memPtr[idx*sz[2] + j*2 + 1] * variance[0] * prior.height) * imgHeight;
        }
    }
    return landmarks;
}

std::vector<ModelRetinaFacePT::Box> ModelRetinaFacePT::generatePriorData() {
    float globalMinSizes[][2] = { {16, 32}, {64, 128}, {256, 512} };
    float steps[] = { 8., 16., 32. };
    std::vector<ModelRetinaFacePT::Box> anchors;
    for (size_t stepNum = 0; stepNum < arraySize(steps); stepNum++) {
        const int featureW = (int)std::round(netInputWidth / steps[stepNum]);
        const int featureH = (int)std::round(netInputHeight / steps[stepNum]);

        auto& minSizes = globalMinSizes[stepNum];
        for (int i = 0; i < featureH; i++) {
            for (int j = 0; j < featureW; j++) {
                for (auto minSize : minSizes) {
                    float sKX = minSize / netInputWidth;
                    float sKY = minSize / netInputHeight;
                    float denseCY = (i + 0.5f) * steps[stepNum] / netInputHeight;
                    float denseCX = (j + 0.5f) * steps[stepNum] / netInputWidth;
                    anchors.push_back(ModelRetinaFacePT::Box{denseCX, denseCY, sKX, sKY});
                }
            }
        }
    }
    return anchors;
}

std::vector<ModelRetinaFacePT::Rect> ModelRetinaFacePT::getFilteredProposals(const InferenceEngine::MemoryBlob::Ptr& rawData, const std::vector<size_t>& indicies,int imgWidth, int imgHeight) {
    std::vector<ModelRetinaFacePT::Rect> rects;
    rects.reserve(indicies.size());

    InferenceEngine::LockedMemory<const void> outputMapped = rawData->rmap();
    const float *memPtr = outputMapped.as<float*>();
    auto desc = rawData->getTensorDesc();
    auto sz = desc.getDims();


    if (sz[1] != priors.size()) {
        throw std::runtime_error("rawBoxes size is not equal to priors size");
    }

    for (auto i : indicies) {
        auto pRawBox = reinterpret_cast<const Box*>(memPtr + i*sz[2]);
        auto& prior = priors[i];
        float cX = priors[i].cX + pRawBox->cX * variance[0] * prior.width;
        float cY = priors[i].cY + pRawBox->cY * variance[0] * prior.width;
        float width = prior.width * exp(pRawBox->width * variance[1]);
        float height = prior.height * exp(pRawBox->height * variance[1]);
        rects.push_back(Rect{
            (cX - width / 2) * imgWidth,
            (cY - height / 2) * imgHeight,
            (cX + width / 2) * imgWidth,
            (cY + height / 2) * imgHeight });
    }

    return rects;
}

std::unique_ptr<ResultBase> ModelRetinaFacePT::postprocess(InferenceResult& infResult) {
    //(raw_output, scale_x, scale_y, face_prob_threshold, image_size):
    const auto bboxRaw = infResult.outputsData[outputsNames[OT_BBOX]];
    const auto scoresRaw = infResult.outputsData[outputsNames[OT_SCORES]];

    const auto& validIndicies = filterByScore(scoresRaw, confidenceThreshold);
    const auto& scores = getFilteredScores(scoresRaw, validIndicies);

    const auto& internalData = infResult.internalModelData->asRef<InternalImageModelData>();
    const auto& landmarks = landmarksNum ?
        getFilteredLandmarks(infResult.outputsData[outputsNames[OT_LANDMARK]], validIndicies, internalData.inputImgWidth, internalData.inputImgHeight) :
        std::vector<cv::Point2f>();

    const auto& proposals = getFilteredProposals(bboxRaw, validIndicies, internalData.inputImgWidth, internalData.inputImgHeight);

    const auto& keptIndicies = nms(proposals, scores, boxIOUThreshold, !landmarksNum);

    // --------------------------- Create detection result objects --------------------------------------------------------
    RetinaFaceDetectionResult* result = new RetinaFaceDetectionResult(infResult.frameId, infResult.metaData);

    result->objects.reserve(keptIndicies.size());
    result->landmarks.reserve(keptIndicies.size() * landmarksNum);
    for (auto i : keptIndicies) {
        DetectedObject desc;
        desc.confidence = scores[i];

        //--- Scaling coordinates
        desc.x = proposals[i].left;
        desc.y = proposals[i].top;
        desc.width = proposals[i].getWidth();
        desc.height = proposals[i].getHeight();

        desc.labelID = 0;
        desc.label = labels[desc.labelID];
        result->objects.push_back(desc);

        //--- Filtering landmarks coordinates
        for (uint32_t l = 0; l < landmarksNum; ++l) {
            result->landmarks.emplace_back(landmarks[i*landmarksNum +l].x,
                landmarks[i*landmarksNum + l].y
            );
        }
    }

    return std::unique_ptr<ResultBase>(result);
}
