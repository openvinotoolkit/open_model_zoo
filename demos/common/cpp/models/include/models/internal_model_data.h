/*
// Copyright (C) 2018-2021 Intel Corporation
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
struct InternalModelData
{
    virtual ~InternalModelData() {}

    template<class T> T& asRef() {
        return dynamic_cast<T&>(*this);
    }

    template<class T> const T& asRef() const {
        return dynamic_cast<const T&>(*this);
    }
};

struct InternalImageModelData : public InternalModelData
{
    InternalImageModelData(int width, int height) :
        inputImgWidth(width),
        inputImgHeight(height) {}

    int inputImgWidth;
    int inputImgHeight;
};

struct InternalImageMatModelData : public InternalImageModelData
{
    InternalImageMatModelData(const cv::Mat& mat) :
        InternalImageModelData(mat.cols, mat.rows), mat(mat) {}

    InternalImageMatModelData(const cv::Mat& mat, int width, int height) :
        InternalImageModelData(width, height), mat(mat) {}

    cv::Mat mat;
};

struct InternalScaleMatData : public InternalModelData
{
    InternalScaleMatData(float scaleX, float scaleY, cv::Mat&& mat) :
        x(scaleX), y(scaleY), mat(std::move(mat)) {}

    float x;
    float y;
    cv::Mat mat;
};
