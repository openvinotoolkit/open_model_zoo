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

#include "models/image_model.h"

ImageModel::ImageModel(const std::string& modelFileName, bool useAutoResize) :
    ModelBase(modelFileName),
    useAutoResize(useAutoResize) {
}

std::shared_ptr<InternalModelData> ImageModel::preprocess(const InputData& inputData, InferenceEngine::InferRequest::Ptr& request) {
    const auto& origImg = inputData.asRef<ImageInputData>().inputImage;
    const auto& img = inputTransform(origImg);

    if (useAutoResize) {
        /* Just set input blob containing read image. Resize and layout conversionx will be done automatically */
        request->SetBlob(inputsNames[0], wrapMat2Blob(img));
    }
    else {
        /* Resize and copy data from the image to the input blob */
        InferenceEngine::Blob::Ptr frameBlob = request->GetBlob(inputsNames[0]);
        matToBlob(img, frameBlob);
    }
    return std::make_shared<InternalImageModelData>(img.cols, img.rows);
}
