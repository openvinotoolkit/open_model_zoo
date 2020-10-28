/*
// Copyright (C) 2018-2020 Intel Corporation
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
#include <samples/ocv_common.hpp>

struct MetaData {
    virtual ~MetaData() {}

    template<class T> T* asPtr(){
        const auto dst = dynamic_cast<T*>(this);
        if (!dst) {
            throw std::bad_cast();
        }
        return dst;
    }

    template<class T> const T* asPtr() const {
        const auto dst = dynamic_cast<const T*>(this);
        if (!dst) {
            throw std::bad_cast();
        }
        return dst;
    }
};

struct ImageMetaData : public MetaData {
    cv::Mat img;

    ImageMetaData() {
    }

    ImageMetaData(cv::Mat img){
        this->img = img;
    }
};
