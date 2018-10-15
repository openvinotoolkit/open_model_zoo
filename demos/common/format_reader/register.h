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
 * \brief Register for readers
 * \file register.h
 */
#pragma once

#include <format_reader.h>
#include <functional>
#include <vector>
#include <string>

namespace FormatReader {
/**
 * \class Registry
 * \brief Create reader from fabric
 */
class Registry {
private:
    typedef std::function<Reader *(const std::string &filename)> CreatorFunction;
    static std::vector<CreatorFunction> _data;
public:
    /**
     * \brief Create reader
     * @param filename - path to input data
     * @return Reader for input data or nullptr
     */
    static Reader *CreateReader(const char *filename);

    /**
     * \brief Registers reader in fabric
     * @param f - a creation function
     */
    static void RegisterReader(CreatorFunction f);
};

/**
 * \class Register
 * \brief Registers reader in fabric
 */
template<typename To>
class Register {
public:
    /**
     * \brief Constructor creates creation function for fabric
     * @return Register object
     */
    Register() {
        Registry::RegisterReader([](const std::string &filename) -> Reader * {
            return new To(filename);
        });
    }
};
}  // namespace FormatReader