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

#ifdef USE_VA
#include "vaapi_images.h"
#endif

#include "utils/uni_image.h"

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
#ifdef USE_VA
    InternalImageModelData(int width, int height, InferenceBackend::VaApiImage::Ptr vaImage=nullptr) :
        vaImage(vaImage),
        inputImgWidth(width),
        inputImgHeight(height)
    {
    }

    InferenceBackend::VaApiImage::Ptr vaImage;
#else
    InternalImageModelData(int width, int height, const UniImage::Ptr& img) :
        inputImgWidth(width),
        inputImgHeight(height),
        img(img) {}
#endif

    int inputImgWidth;
    int inputImgHeight;
    UniImage::Ptr img;
};

struct InternalScaleMatData : public InternalModelData {
    InternalScaleMatData(float scaleX, float scaleY, const UniImage::Ptr& img) :
        x(scaleX), y(scaleY), img(img) {}

    float x;
    float y;
    UniImage::Ptr img;
};
