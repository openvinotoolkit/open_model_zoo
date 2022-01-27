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
//class StyleTransferModel : public ImageModel {
//public:
//    /// Constructor
//    /// @param modelFileName name of model to load
//    StyleTransferModel(const std::string& modelFileName);
//
//    std::shared_ptr<InternalModelData> preprocess(
//        const InputData& inputData, ov::InferRequest& request) override;
//    std::unique_ptr<ResultBase> postprocess(InferenceResult& infResult) override;
//
//protected:
//    void prepareInputsOutputs(std::shared_ptr<ov::Model> & model) override;
//};
