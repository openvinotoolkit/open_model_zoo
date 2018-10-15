/*
// Copyright (c) 2018 Intel Corporation
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

/**
 * \brief Image reader
 * \file opencv_wraper.h
 */
#pragma once

#ifdef USE_OPENCV
#include <memory>
#include <string>
#include <format_reader.h>

#include <opencv2/opencv.hpp>

#include "register.h"

namespace FormatReader {
/**
 * \class OCVMAT
 * \brief OpenCV Wraper
 */
class OCVReader : public Reader {
private:
    cv::Mat img;
    size_t _size;
    static Register<OCVReader> reg;

public:
    /**
    * \brief Constructor of BMP reader
    * @param filename - path to input data
    * @return BitMap reader object
    */
    explicit OCVReader(const std::string &filename);
    virtual ~OCVReader() {
    }

    /**
    * \brief Get size
    * @return size
    */
    size_t size() const override {
        return _size;
    }

    void Release() noexcept override {
        delete this;
    }

    std::shared_ptr<unsigned char> getData(int width, int height) override;
};
}  // namespace FormatReader
#endif