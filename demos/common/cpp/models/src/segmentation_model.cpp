///*
//// Copyright (C) 2018-2021 Intel Corporation
////
//// Licensed under the Apache License, Version 2.0 (the "License");
//// you may not use this file except in compliance with the License.
//// You may obtain a copy of the License at
////
////      http://www.apache.org/licenses/LICENSE-2.0
////
//// Unless required by applicable law or agreed to in writing, software
//// distributed under the License is distributed on an "AS IS" BASIS,
//// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//// See the License for the specific language governing permissions and
//// limitations under the License.
//*/
//
//#include "models/segmentation_model.h"
//#include "utils/ocv_common.hpp"
//
//SegmentationModel::SegmentationModel(const std::string& modelFileName, bool useAutoResize) :
//    ImageModel(modelFileName, useAutoResize) {}
//
//std::vector<std::string> SegmentationModel::loadLabels(const std::string & labelFilename)
//{
//    std::vector<std::string> labelsList;
//
//    /* Read labels (if any) */
//    if (!labelFilename.empty()) {
//        std::ifstream inputFile(labelFilename);
//        if (!inputFile.is_open())
//            throw std::runtime_error("Can't open the labels file: " + labelFilename);
//        std::string label;
//        while (std::getline(inputFile, label)) {
//            labelsList.push_back(label);
//        }
//        if (labelsList.empty())
//            throw std::logic_error("File is empty: " + labelFilename);
//    }
//
//    return labelsList;
//}
//
//void SegmentationModel::prepareInputsOutputs(std::shared_ptr<ov::Model>& model)
//{
//    // --------------------------- Configure input & output ---------------------------------------------
//    // --------------------------- Prepare input blobs -----------------------------------------------------
//    const ov::OutputVector& inputsInfo = model->inputs();
//    if (inputShapes.size() != 1) {
//        throw std::runtime_error("Demo supports topologies only with 1 input");
//    }
//
//    inputsNames.push_back(inputShapes.begin()->first);
//
//    const ov::Shape& inputShape = model->input().get_shape();
//    if (inSizeVector.size() != 4 || inSizeVector[1] != 3)
//        throw std::runtime_error("3-channel 4-dimensional model's input is expected");
//
//    InferenceEngine::InputInfo& inputInfo = *model.getInputsInfo().begin()->second;
//    inputInfo.setPrecision(InferenceEngine::Precision::U8);
//
//    if (useAutoResize) {
//        inputInfo.getPreProcess().setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
//        inputInfo.setLayout(InferenceEngine::Layout::NHWC);
//    } else {
//        inputInfo.setLayout(InferenceEngine::Layout::NCHW);
//    }
//    // --------------------------- Prepare output blobs -----------------------------------------------------
//    const InferenceEngine::OutputsDataMap& outputsDataMap = model.getOutputsInfo();
//    if (outputsDataMap.size() != 1) throw std::runtime_error("Demo supports topologies only with 1 output");
//
//    outputsNames.push_back(outputsDataMap.begin()->first);
//    InferenceEngine::Data& data = *outputsDataMap.begin()->second;
//
//    const InferenceEngine::SizeVector& outSizeVector = data.getTensorDesc().getDims();
//    switch (outSizeVector.size()) {
//    case 3:
//        outChannels = 1;
//        outHeight = (int)(outSizeVector[1]);
//        outWidth = (int)(outSizeVector[2]);
//        break;
//    case 4:
//        outChannels = (int)(outSizeVector[1]);
//        outHeight = (int)(outSizeVector[2]);
//        outWidth = (int)(outSizeVector[3]);
//        break;
//    default:
//        throw std::runtime_error("Unexpected output blob shape. Only 4D and 3D output blobs are supported.");
//    }
//}
//
//std::unique_ptr<ResultBase> SegmentationModel::postprocess(InferenceResult& infResult) {
//    ImageResult* result = new ImageResult(infResult.frameId, infResult.metaData);
//
//    const auto& inputImgSize = infResult.internalModelData->asRef<InternalImageModelData>();
//
//    InferenceEngine::MemoryBlob::Ptr blobPtr = infResult.getFirstOutputBlob();
//
//    void* pData = blobPtr->rmap().as<void*>();
//
//    result->resultImage = cv::Mat(outHeight, outWidth, CV_8UC1);
//
//    if (outChannels == 1 && blobPtr->getTensorDesc().getPrecision() == InferenceEngine::Precision::I32)
//    {
//        cv::Mat predictions(outHeight, outWidth, CV_32SC1, pData);
//        predictions.convertTo(result->resultImage, CV_8UC1);
//    }
//    else if (blobPtr->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP32)
//    {
//        float* ptr = reinterpret_cast<float*>(pData);
//        for (int rowId = 0; rowId < outHeight; ++rowId)
//        {
//            for (int colId = 0; colId < outWidth; ++colId)
//            {
//                int classId = 0;
//                float maxProb = -1.0f;
//                for (int chId = 0; chId < outChannels; ++chId)
//                {
//                    float prob = ptr[chId * outHeight * outWidth + rowId * outWidth + colId];
//                    if (prob > maxProb)
//                    {
//                        classId = chId;
//                        maxProb = prob;
//                    }
//                } // nChannels
//
//                result->resultImage.at<uint8_t>(rowId, colId) = classId;
//            } // width
//        } // height
//    }
//
//    cv::resize(result->resultImage, result->resultImage,
//        cv::Size(inputImgSize.inputImgWidth, inputImgSize.inputImgHeight),
//        0, 0, cv::INTER_NEAREST);
//
//    return std::unique_ptr<ResultBase>(result);
//}
