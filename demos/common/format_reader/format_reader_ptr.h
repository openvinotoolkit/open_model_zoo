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
 * \brief Implementation of smart pointer for Reader class
 * \file format_reader_ptr.h
 */
#pragma once

#include "format_reader.h"
#include <functional>
#include <memory>

namespace FormatReader {
class ReaderPtr {
public:
    explicit ReaderPtr(const char *imageName) : reader(CreateFormatReader(imageName),
                                                [](Reader *p) {
                                                p->Release();
                                           }) {}
    /**
     * @brief dereference operator overload
     * @return Reader
     */
    Reader *operator->() const noexcept {
        return reader.get();
    }

    /**
     * @brief dereference operator overload
     * @return Reader
     */
    Reader *operator*() const noexcept {
        return reader.get();
    }

    Reader *get() {
        return reader.get();
    }

protected:
    std::unique_ptr<Reader, std::function<void(Reader *)>> reader;
};
}  // namespace FormatReader
