/*
// Copyright (C) 2021 Intel Corporation
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

#include "models/detection_model.h"
#include <utils/ocv_common.hpp>
#include <utils/slog.hpp>

using namespace InferenceEngine;

ImageModel::ImageModel(const std::string& modelFileName, bool useAutoResize) :
    ModelBase(modelFileName),
    useAutoResize(useAutoResize) {
}

InferenceEngine::ExecutableNetwork ImageModel::loadExecutableNetwork(const CnnConfig & cnnConfig, InferenceEngine::Core & core) {
    this->cnnConfig = cnnConfig;
    auto cnnNetwork = prepareNetwork(core);
    execNetwork = core.LoadNetwork(cnnNetwork, cnnConfig.devices, cnnConfig.execNetworkConfig);
    return execNetwork;
}

std::shared_ptr<InternalModelData> ImageModel::preprocess(const InputData& inputData, InferenceEngine::InferRequest::Ptr& request) {
    auto& data = inputData.asRef<ImageInputData>();
    int width = 0;
    int height = 0;

    auto& img = data.inputImage;
    width = img.cols;
    height = img.rows;

    if (useAutoResize) {
        /* Just set input blob containing read image. Resize and layout conversionx will be done automatically */
        request->SetBlob(inputsNames[0], wrapMat2Blob(img));
        /* IE::Blob::Ptr from wrapMat2Blob() doesn't own data. Save the image to avoid deallocation before inference */
        return std::make_shared<InternalImageMatModelData>(img);
    }
    /* Resize and copy data from the image to the input blob */
    Blob::Ptr frameBlob = request->GetBlob(inputsNames[0]);
    matU8ToBlob<uint8_t>(img, frameBlob);
    return std::make_shared<InternalImageModelData>(img.cols, img.rows);
}
