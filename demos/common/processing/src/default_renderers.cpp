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

#include <default_renderers.h>
#include <opencv2/imgproc.hpp>
#include <iostream>

namespace DefaultRenderers {
    cv::Mat renderDetectionData(const DetectionResult& result)
    {
        if (!result.metaData) {
            throw std::invalid_argument("Renderer: metadata is null");
        }

        auto outputImg = result.metaData->asRef<ImageMetaData>().img.clone();

        if (outputImg.empty()) {
            throw std::invalid_argument("Renderer: image provided in metadata is empty");
        }

        // Visualizing result data over source image

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

    cv::Mat renderSegmentationData(const SegmentationResult& result){
        if (!result.metaData){
            throw std::invalid_argument("Renderer: metadata is null");
        }

        auto inputImg = result.metaData->asRef<ImageMetaData>().img;

        if (inputImg.empty()) {
            throw std::invalid_argument("Renderer: image provided in metadata is empty");
        }

        // Visualizing result data over source image
        cv::Mat outputImg;
        if (inputImg.size != result.mask.size) {
            cv::resize(result.mask, outputImg, inputImg.size());
            outputImg = inputImg / 2 + outputImg / 2;
        }
        else {
            outputImg = inputImg / 2 + result.mask / 2;
        }

        return outputImg;
    }
}
