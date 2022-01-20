///*
//// Copyright (C) 2021 Intel Corporation
////
//// Licensed under the Apache License, Version 2.0 (the "License");
//// you may not use this file except in compliance with the License.
//// You may obtain a copy of the License at
////
////      http://www.apache.org/licenses/LICENSE-2.0
////
//// Unless required by applicable law or agreed to in writingb  software
//// distributed under the License is distributed on an "AS IS" BASIS,
//// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//// See the License for the specific language governing permissions and
//// limitations under the License.
//*/
//
//#pragma once
//
//#include "image_model.h"
//
//class JPEGRestorationModel : public ImageModel {
//public:
//    /// Constructor
//    /// @param modelFileName name of model to load
//    /// @param inputImgSize size of image to set model input shape
//    /// @param jpegCompression flag allows to perform compression before the inference
//    JPEGRestorationModel(const std::string& modelFileName, const cv::Size& inputImgSize, bool jpegCompression);
//
//    std::shared_ptr<InternalModelData> preprocess(
//        const InputData& inputData, InferenceEngine::InferRequest::Ptr& request) override;
//    std::unique_ptr<ResultBase> postprocess(InferenceResult& infResult) override;
//
//protected:
//    void prepareInputsOutputs(InferenceEngine::CNNNetwork & cnnNetwork) override;
//    void changeInputSize(InferenceEngine::CNNNetwork& cnnNetwork);
//
//    static const size_t stride = 8;
//    bool jpegCompression = false;
//};
