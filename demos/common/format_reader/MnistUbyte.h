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
 * \brief Mnist reader
 * \file MnistUbyte.h
 */
#pragma once

#include <memory>
#include <string>
#include <format_reader.h>

#include "register.h"

namespace FormatReader {
/**
 * \class MnistUbyte
 * \brief Reader for mnist db files
 */
class MnistUbyte : public Reader {
private:
    int reverseInt(int i);

    static Register<MnistUbyte> reg;

public:
    /**
     * \brief Constructor of Mnist reader
     * @param filename - path to input data
     * @return MnistUbyte reader object
     */
    explicit MnistUbyte(const std::string &filename);
    virtual ~MnistUbyte() {
    }

    /**
     * \brief Get size
     * @return size
     */
    size_t size() const override {
        return _width * _height * 1;
    }

    void Release() noexcept override {
        delete this;
    }

    std::shared_ptr<unsigned char> getData(int width, int height) override {
        if ((width * height != 0) && (_width * _height != width * height)) {
            std::cout << "[ WARNING ] Image won't be resized! Please use OpenCV.\n";
            return nullptr;
        }
        return _data;
    }
};
}  // namespace FormatReader
