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
#include <samples/slog.hpp>

#include "detection_pipeline_yolo.h"
#include "detection_pipeline_ssd.h"

using namespace InferenceEngine;

void DetectionPipeline::init(const std::string& model_name, const CnnConfig& cnnConfig,
    float confidenceThreshold, bool useAutoResize,
    const std::vector<std::string>& labels,
    InferenceEngine::Core* engine){

    this->useAutoResize = useAutoResize;
    this->confidenceThreshold = confidenceThreshold;
    this->labels = labels;

    PipelineBase::init(model_name, cnnConfig, engine);
}

int64_t DetectionPipeline::submitImage(cv::Mat img){
    auto request = requestsPool->getIdleRequest();
    if (!request)
        return -1;

    if (useAutoResize) {
        /* Just set input blob containing read image. Resize and layout conversionx will be done automatically */
        request->SetBlob(imageInputName, wrapMat2Blob(img));
    }
    else {
        /* Resize and copy data from the image to the input blob */
        Blob::Ptr frameBlob = request->GetBlob(imageInputName);
        matU8ToBlob<uint8_t>(img, frameBlob);
    }

    return submitRequest(request,img);
}

cv::Mat DetectionPipeline::obtainAndRenderData()
{
    DetectionResult result = getProcessedResult();
    if (result.IsEmpty()) {
        return cv::Mat();
    }

    // Visualizing result data over source image
    cv::Mat outputImg = result.extraData.clone();

    for (auto obj : result.objects) {
        std::ostringstream conf;
        conf << ":" << std::fixed << std::setprecision(3) << obj.confidence;
        cv::putText(outputImg, obj.label + conf.str(),
            cv::Point2f(obj.x, obj.y - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1,
            cv::Scalar(0, 0, 255));
        cv::rectangle(outputImg, obj, cv::Scalar(0, 0, 255));
    }

    return outputImg;
}

std::vector<std::string> DetectionPipeline::loadLabels(const std::string & labelFilename){
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
