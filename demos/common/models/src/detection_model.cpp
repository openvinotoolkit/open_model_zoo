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

#include "models/detection_model.h"
#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>

using namespace InferenceEngine;

DetectionModel::DetectionModel(const std::string& modelFileName, float confidenceThreshold, bool useAutoResize, const std::vector<std::string>& labels) :
    ModelBase(modelFileName),
    labels(labels),
    useAutoResize(useAutoResize),
    confidenceThreshold(confidenceThreshold) {
}

std::shared_ptr<InternalModelData> DetectionModel::preprocess(const InputData& inputData, InferenceEngine::InferRequest::Ptr& request) {
    auto& img = inputData.asRef<ImageInputData>().inputImage;

    if (useAutoResize) {
        /* Just set input blob containing read image. Resize and layout conversionx will be done automatically */
        request->SetBlob(inputsNames[0], wrapMat2Blob(img));
    }
    else {
        /* Resize and copy data from the image to the input blob */
        Blob::Ptr frameBlob = request->GetBlob(inputsNames[0]);
        matU8ToBlob<uint8_t>(img, frameBlob);
    }

    return std::shared_ptr<InternalModelData>(new InternalImageModelData(img.cols, img.rows));
}

std::vector<std::string> DetectionModel::loadLabels(const std::string& labelFilename) {
    std::vector<std::string> labelsList;

    /** Read labels (if any)**/
    if (!labelFilename.empty()) {
        std::ifstream inputFile(labelFilename);
        std::string label;
        while (std::getline(inputFile, label)) {
            labelsList.push_back(label);
        }
        if (labelsList.empty())
            throw std::logic_error("File empty or not found: " + labelFilename);
    }

    return labelsList;
}
