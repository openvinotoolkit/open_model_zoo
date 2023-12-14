/*
// Copyright (C) 2018-2023 Intel Corporation
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

#pragma once
#include <utils/ocv_common.hpp>

struct MetaData {
    virtual ~MetaData() {}

    template <class T>
    T& asRef() {
        return dynamic_cast<T&>(*this);
    }

    template <class T>
    const T& asRef() const {
        return dynamic_cast<const T&>(*this);
    }
};

struct ImageMetaData : public MetaData {
    cv::Mat img;
    std::chrono::steady_clock::time_point timeStamp;

    ImageMetaData() {}

    ImageMetaData(cv::Mat img, std::chrono::steady_clock::time_point timeStamp) : img(img), timeStamp(timeStamp) {}
};

struct ImageBatchMetaData : public MetaData {
    std::chrono::steady_clock::time_point timeStamp;
    std::vector<std::shared_ptr<ImageMetaData>> metadatas;

    ImageBatchMetaData() {}

    ImageBatchMetaData(std::vector<cv::Mat>::iterator imagesBeginIt,
                       const std::vector<cv::Mat>::iterator imagesEndIt,
                       std::chrono::steady_clock::time_point timeStamp) : timeStamp(timeStamp) {
        size_t images_count = std::distance(imagesBeginIt, imagesEndIt);
        metadatas.reserve(images_count);
        for (; imagesBeginIt != imagesEndIt;) {
            metadatas.push_back(std::make_shared<ImageMetaData>(*imagesBeginIt++, timeStamp));
        }
    }

    void add(cv::Mat img, std::chrono::steady_clock::time_point timeStamp) {
        metadatas.push_back(std::make_shared<ImageMetaData>(img, timeStamp));
        this->timeStamp = timeStamp;
    }
    void clear() {
        metadatas.clear();
    }
};

struct ClassificationImageMetaData : public ImageMetaData {
    unsigned int groundTruthId;

    ClassificationImageMetaData(cv::Mat img,
                                std::chrono::steady_clock::time_point timeStamp,
                                unsigned int groundTruthId)
        : ImageMetaData(img, timeStamp),
          groundTruthId(groundTruthId) {}
};


struct ClassificationImageBatchMetaData : public MetaData {
    std::vector<std::shared_ptr<ClassificationImageMetaData>> metadatas;

    ClassificationImageBatchMetaData(std::vector<cv::Mat>::iterator imagesBeginIt,
                                const std::vector<cv::Mat>::iterator imagesEndIt,
                                std::chrono::steady_clock::time_point timeStamp,
                                std::vector<unsigned int>::iterator groundTruthIdsBeginIt,
                                const std::vector<unsigned int>::iterator groundTruthIdsEndIt)
        : MetaData(){
        size_t images_count = std::distance(imagesBeginIt, imagesEndIt);
        size_t gt_count = std::distance(groundTruthIdsBeginIt, groundTruthIdsEndIt);
        if (images_count != gt_count) {
            throw std::runtime_error("images.size() != groundTruthIds.size()");
        }

        metadatas.reserve(images_count);
        for (; imagesBeginIt != imagesEndIt;) {
            metadatas.push_back(std::make_shared<ClassificationImageMetaData>(*imagesBeginIt++, timeStamp, *groundTruthIdsBeginIt++));
        }
    }
};
