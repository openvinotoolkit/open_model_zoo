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
#include "utils.hpp"

#include <algorithm>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <unistd.h>

namespace mcam {

string_ref::string_ref(const char* str):
    ptr(str),
    len(nullptr != str ? std::strlen(str) : 0) {
}

string_ref::string_ref(const std::string& str):
    ptr(!str.empty() ? str.c_str() : nullptr),
    len(str.size()) {
}

bool operator==(const string_ref& str, std::nullptr_t) {
    return str.empty();
}

bool operator==(std::nullptr_t, const string_ref& str) {
    return str.empty();
}

bool operator!=(const string_ref& str, std::nullptr_t) {
    return !str.empty();
}

bool operator!=(std::nullptr_t, const string_ref& str) {
    return !str.empty();
}

[[noreturn]] void throw_errno_error(string_ref desc, int err) {
    std::stringstream ss;
    if (nullptr != desc) {
        ss << desc.data();
    }
    ss << " " << err << ", " << strerror(err);
    throw std::logic_error(ss.str());
}

void throw_error(string_ref desc) {
    throw std::logic_error(desc == nullptr ? "Generic error" : desc.data());
}

file_descriptor::file_descriptor(int fd_):
    desc(fd_) {
}

file_descriptor::~file_descriptor() {
    if (-1 != desc) {
        close(desc);
    }
}

file_descriptor& file_descriptor::operator=(file_descriptor&& other) {
    if (this != &other) {
        swap(other);
    }
    return *this;
}

void file_descriptor::swap(file_descriptor& other) {
    std::swap(desc, other.desc);
}

}  // namespace mcam
